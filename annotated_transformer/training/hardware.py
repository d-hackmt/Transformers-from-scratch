"""Helpers for running training on a machine that might overheat or lose power.

Nothing here changes the model. It exists because sustained GPU + CPU load can
trip a marginal power supply or a hot VRM, and because a checkpoint written in
place is corrupted if the machine dies mid-write.

* :func:`gpu_stats`    - read temperature / power / memory from ``nvidia-smi``
* :func:`save_atomic`  - crash-safe checkpoint: write to a temp file, fsync,
  atomically rename, and keep the previous version as a fallback
* :func:`cooldown`     - pause between epochs so thermals recover, watching the
  GPU temperature and aborting gracefully if it climbs too high
"""

import os
import shutil
import subprocess
import time


def gpu_stats():
    """Return ``{"temp_c", "power_w", "mem_mb", "util_pct"}`` or ``None``.

    Shells out to ``nvidia-smi``; returns ``None`` if it is missing or fails
    (e.g. CPU-only machine), so callers can treat GPU monitoring as optional.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        temp, power, mem, util = (x.strip() for x in out.strip().splitlines()[0].split(","))
        return {
            "temp_c": float(temp),
            "power_w": float(power),
            "mem_mb": float(mem),
            "util_pct": float(util),
        }
    except Exception:
        return None


def save_atomic(obj, path: str, keep_prev: bool = True) -> None:
    """Save ``obj`` (via ``torch.save``) so a crash never leaves a corrupt file.

    Steps: write ``<path>.tmp`` -> flush + fsync -> rotate the existing file to
    ``<path>.prev`` -> atomically rename ``.tmp`` onto ``path``. If the machine
    dies at any point, either the old ``path``, ``path.prev``, or a stray
    ``.tmp`` survives - never a half-written ``path``.
    """
    import torch

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if keep_prev and os.path.exists(path):
        try:
            shutil.copy2(path, path + ".prev")
        except OSError:
            pass
    os.replace(tmp, path)  # atomic on Windows and POSIX


class CooldownAbort(Exception):
    """Raised by :func:`cooldown` when the GPU is too hot to continue safely."""


def cooldown(seconds: int, log_fn=print, temp_limit: int = 80, temp_abort: int = 90) -> None:
    """Sleep ``seconds`` between epochs, watching GPU temperature.

    * logs the current GPU temp / power at the start of the pause;
    * if temp >= ``temp_limit`` the pause is doubled (once);
    * if temp >= ``temp_abort`` raises :class:`CooldownAbort` so the caller can
      checkpoint and exit rather than risk a hard power-off.

    With no GPU (``nvidia-smi`` unavailable) it just sleeps.
    """
    if seconds <= 0:
        return
    stats = gpu_stats()
    if stats:
        log_fn(
            f"cooldown: GPU {stats['temp_c']:.0f} C, {stats['power_w']:.0f} W, "
            f"{stats['mem_mb']:.0f} MiB — pausing {seconds}s"
        )
        if stats["temp_c"] >= temp_abort:
            raise CooldownAbort(f"GPU at {stats['temp_c']:.0f} C (>= {temp_abort})")
        if stats["temp_c"] >= temp_limit:
            seconds *= 2
            log_fn(f"cooldown: GPU warm ({stats['temp_c']:.0f} C) — extending pause to {seconds}s")
    else:
        log_fn(f"cooldown: pausing {seconds}s")

    # sleep in short slices so the process stays responsive to Ctrl-C
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(min(5, end - time.time()))
