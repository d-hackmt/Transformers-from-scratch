"""Live training monitor: a rolling text log + an auto-refreshing HTML dashboard.

``TrainingMonitor`` is created by ``scripts/run_experiments.py``. It:

* appends timestamped lines to ``<out>/train.log`` (also echoed to stdout), so you
  can ``tail -f`` the run;
* after every progress tick and every epoch, rewrites ``<out>/live.html`` - a
  single self-contained file (no external assets) that refreshes itself every few
  seconds. Open it in a browser and watch the numbers move.

Nothing here is required for training; it is pure observability.
"""

import html
import os
import re
import time

from annotated_transformer.training.hardware import gpu_stats

# matches the progress line run_epoch emits (kept in the notebook's wording)
_STEP_RE = re.compile(
    r"Epoch Step:\s*(\d+).*?Loss:\s*([\d.]+).*?Tokens / Sec:\s*([\d.]+).*?Learning Rate:\s*([\d.eE+-]+)"
)

# GPU temperature bands used for the dashboard colours (°C).
TEMP_OK = 70      # green below this
TEMP_WARM = 80    # amber
TEMP_HOT = 87     # red + banner  ("tell me to stop")


def _temp_color(t):
    if t is None:
        return "var(--muted)"
    if t < TEMP_OK:
        return "#22c55e"
    if t < TEMP_WARM:
        return "#f59e0b"
    if t < TEMP_HOT:
        return "#f97316"
    return "#ef4444"


class TrainingMonitor:
    def __init__(self, out_dir: str, total_epochs: int, refresh_seconds: int = 10,
                 fresh: bool = True, steps_per_epoch: int = None):
        self.out_dir = out_dir
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.refresh_seconds = refresh_seconds
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "train.log")
        self.html_path = os.path.join(out_dir, "live.html")
        self.start = time.time()
        self.current_epoch = 0
        self.status_line = "starting ..."
        self.gpu = None            # latest {"temp_c","power_w",...} or None
        self.gpu_peak = 0.0        # hottest reading seen this session
        self._last_gpu_poll = 0.0
        self._metrics = {"epochs": [], "config": {}, "checkpoint_epochs": []}
        # truncate the log on a fresh run; append when resuming
        with open(self.log_path, "w" if fresh else "a", encoding="utf-8") as f:
            tag = "started" if fresh else "resumed"
            f.write(f"# training log — {tag} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ------------------------------------------------------------------ logging
    def _poll_gpu(self):
        """Refresh the cached GPU reading at most every ~12 s (cheap subprocess)."""
        now = time.time()
        if now - self._last_gpu_poll < 12:
            return
        self._last_gpu_poll = now
        s = gpu_stats()
        if s:
            self.gpu = s
            self.gpu_peak = max(self.gpu_peak, s["temp_c"])

    def log(self, msg: str) -> None:
        """Timestamped line to train.log + stdout, then refresh the dashboard."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        # progress ticks from run_epoch look like "Epoch Step: ... | Loss: ..."
        m = _STEP_RE.search(msg)
        if m:
            batch, loss, tps, lr = m.groups()
            total = f"/{self.steps_per_epoch}" if self.steps_per_epoch else ""
            self.status_line = (
                f"Epoch {self.current_epoch}/{self.total_epochs}  ·  "
                f"batch {batch}{total}  ·  loss {loss}  ·  {float(tps):.0f} tok/s  ·  lr {lr}"
            )
        self._poll_gpu()  # rate-limited internally to ~12s
        self.render(self._metrics)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.status_line = f"epoch {epoch}/{self.total_epochs} — training ..."

    # ---------------------------------------------------------------- dashboard
    def render(self, metrics: dict) -> None:
        """Write live.html from the metrics collected so far."""
        self._metrics = metrics
        try:
            with open(self.html_path, "w", encoding="utf-8") as f:
                f.write(self._html(metrics))
        except OSError:
            pass  # never let a dashboard write kill training

    # ------------------------------------------------------------------ helpers
    def _elapsed_eta(self, metrics: dict):
        rows = metrics.get("epochs", [])
        elapsed = time.time() - self.start
        done = len(rows)
        if done == 0:
            return elapsed, None
        per_epoch = sum(r["seconds"] for r in rows) / done
        remaining = max(self.total_epochs - done, 0) * per_epoch
        return elapsed, remaining

    def _html(self, metrics: dict) -> str:
        rows = metrics.get("epochs", [])
        cfg = metrics.get("config", {})
        elapsed, eta = self._elapsed_eta(metrics)
        done = len(rows)
        pct = int(100 * done / max(self.total_epochs, 1))
        eta_str = _fmt(eta) if eta is not None else "—"
        last_ppl = f"{rows[-1]['val_perplexity']:.2f}" if rows else "—"
        last_acc = f"{rows[-1]['val_accuracy'] * 100:.1f}%" if rows and "val_accuracy" in rows[-1] else "—"
        finished = "total_minutes" in metrics
        checkpoints = [r for r in rows if "bleu_valid" in r]
        best_bleu = max((r["bleu_test"] for r in checkpoints), default=None)
        best_bleu_str = f"{best_bleu:.2f}" if best_bleu is not None else "—"

        # --- GPU temperature -------------------------------------------
        gpu_now = self.gpu["temp_c"] if self.gpu else None
        gpu_power = self.gpu["power_w"] if self.gpu else None
        gpu_color = _temp_color(gpu_now)
        gpu_temp_str = f"{gpu_now:.0f}&deg;C" if gpu_now is not None else "—"
        gpu_sub = (
            f"{gpu_power:.0f} W &nbsp;|&nbsp; peak {self.gpu_peak:.0f}&deg;C"
            if gpu_power is not None else "no reading"
        )
        temp_banner = ""
        if gpu_now is not None and gpu_now >= TEMP_HOT:
            temp_banner = (
                f"<div class='alert'>⚠ GPU at {gpu_now:.0f}&deg;C — this is the danger zone. "
                f"Tell Claude to stop the run.</div>"
            )
        elif gpu_now is not None and gpu_now >= TEMP_WARM:
            temp_banner = (
                f"<div class='warn'>GPU at {gpu_now:.0f}&deg;C — running warm. Keep an eye on it.</div>"
            )
        temp_chart = _svg_chart(
            [("GPU temp °C", [(r["epoch"], r["gpu_temp_c"]) for r in rows if "gpu_temp_c" in r], "#ef4444")],
            y_label="GPU temp °C",
        )

        loss_chart = _svg_chart(
            [
                ("train loss", [(r["epoch"], r["train_loss"]) for r in rows], "#4f8cff"),
                ("val loss (KL)", [(r["epoch"], r["val_loss"]) for r in rows], "#ff5c8a"),
            ],
            y_label="loss / token",
        )
        ppl_chart = _svg_chart(
            [("val perplexity", [(r["epoch"], r["val_perplexity"]) for r in rows], "#22c55e")],
            y_label="perplexity",
        )
        acc_chart = _svg_chart(
            [("token accuracy", [(r["epoch"], r["val_accuracy"] * 100) for r in rows if "val_accuracy" in r], "#f59e0b")],
            y_label="accuracy %",
        )
        kl_chart = _svg_chart(
            [("KL(target ‖ model)", [(r["epoch"], r["kl_snapshot"]) for r in rows if "kl_snapshot" in r], "#a855f7")],
            y_label="KL divergence",
        )
        dist_section = _distribution_section(metrics.get("distribution"))

        # BLEU summary — one card per checkpoint, filled in as BLEU is computed.
        if checkpoints:
            bleu_cards = "".join(
                "<div class='card'>"
                f"<div class='k'>{r['epoch']} epochs</div>"
                f"<div class='v' style='color:#22c55e'>{r['bleu_test']:.2f}</div>"
                f"<div class='k'>test &nbsp;|&nbsp; valid {r['bleu_valid']:.2f} &nbsp;|&nbsp; ppl {r['val_perplexity']:.1f}</div>"
                "</div>"
                for r in checkpoints
            )
            bleu_section = (
                "<h2>BLEU &amp; perplexity by checkpoint</h2>"
                f"<div class='cards'>{bleu_cards}</div>"
            )
            if checkpoints[0].get("bleu_signature"):
                bleu_section += (
                    f"<div class='sub' style='margin-top:8px'>sacreBLEU: "
                    f"<code>{html.escape(checkpoints[0]['bleu_signature'])}</code></div>"
                )
        else:
            bleu_section = (
                "<h2>BLEU &amp; perplexity by checkpoint</h2>"
                "<div class='sub'>BLEU is scored at the checkpoint epochs "
                f"({', '.join(map(str, metrics.get('checkpoint_epochs', [])))}) — "
                "not computed yet.</div>"
            )

        def _cell(r, key, fmt):
            return format(r[key], fmt) if key in r else ""

        metric_rows = "".join(
            "<tr>"
            f"<td>{r['epoch']}</td>"
            f"<td>{r['train_loss']:.4f}</td>"
            f"<td>{r['val_loss']:.4f}</td>"
            f"<td>{r['val_perplexity']:.2f}</td>"
            f"<td>{_cell(r, 'val_accuracy', '.1%')}</td>"
            f"<td>{_cell(r, 'kl_snapshot', '.3f')}</td>"
            f"<td>{_cell(r, 'bleu_valid', '.2f')}</td>"
            f"<td>{_cell(r, 'bleu_test', '.2f')}</td>"
            f"<td>{r['seconds']:.0f}s</td>"
            "</tr>"
            for r in rows
        ) or "<tr><td colspan='9'>no epochs completed yet</td></tr>"

        samples = metrics.get("samples", {})
        sample_html = ""
        if samples:
            latest = sorted(samples.keys(), key=int)[-1]
            items = "".join(
                "<div class='s'>"
                f"<div class='de'>{html.escape(s['src'])}</div>"
                f"<div class='ref'>ref: {html.escape(s['ref'])}</div>"
                f"<div class='hyp'>out: {html.escape(s['hyp'])}</div>"
                "</div>"
                for s in samples[latest]
            )
            sample_html = (
                f"<h2>Sample translations — {latest} epochs</h2>"
                f"<div class='samples'>{items}</div>"
            )

        cfg_html = " · ".join(
            f"{k}={v}" for k, v in cfg.items()
            if k in ("device", "n_layers", "d_model", "d_ff", "heads", "batch_size", "seed")
        )

        direction = cfg.get("direction", "de-en").replace("->", "-").replace("_", "-")
        names = {"de": "German", "en": "English"}
        s, t = (direction.split("-") + ["en"])[:2]
        pair_title = f"{names.get(s, s)} → {names.get(t, t)}"

        return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{self.refresh_seconds}">
<title>Live training — Multi30k {direction}</title>
<style>
  :root {{ --bg:#ffffff; --fg:#1a1a2e; --muted:#6b7280; --card:#f4f5f7; --line:#e5e7eb; --accent:#4f8cff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1117; --fg:#e6e7ea; --muted:#9aa0aa; --card:#181b23; --line:#2a2f3a; --accent:#4f8cff; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--fg);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:28px 0 10px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
  .sub {{ color:var(--muted); margin-bottom:20px; }}
  .cards {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; min-width:130px; }}
  .card .k {{ color:var(--muted); font-size:12px; }}
  .card .v {{ font-size:20px; font-weight:600; margin-top:2px; }}
  .bar {{ height:8px; background:var(--line); border-radius:6px; overflow:hidden; margin:14px 0; }}
  .bar > i {{ display:block; height:100%; background:var(--accent); width:{pct}%; }}
  .status {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; background:var(--card);
            border:1px solid var(--line); border-radius:8px; padding:10px 12px; overflow-x:auto; white-space:nowrap; }}
  table {{ border-collapse:collapse; width:100%; margin-top:6px; }}
  th,td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); }}
  th:first-child,td:first-child {{ text-align:left; }}
  .charts {{ display:flex; gap:20px; flex-wrap:wrap; }}
  .charts > div {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }}
  .samples {{ display:grid; gap:10px; }}
  .s {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
  .s .de {{ font-weight:600; }}
  .s .ref {{ color:var(--muted); }}
  .s .hyp {{ color:var(--accent); }}
  .done {{ background:#22c55e1a; border:1px solid #22c55e; color:var(--fg);
          border-radius:10px; padding:12px 16px; margin-bottom:16px; font-weight:600; }}
  .alert {{ background:#ef44441f; border:1px solid #ef4444; color:var(--fg);
           border-radius:10px; padding:14px 16px; margin-bottom:16px; font-weight:700; font-size:15px; }}
  .warn {{ background:#f59e0b1f; border:1px solid #f59e0b; color:var(--fg);
          border-radius:10px; padding:12px 16px; margin-bottom:16px; font-weight:600; }}
  .card.temp .v {{ color:{gpu_color}; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
</style></head><body>
<h1>Multi30k · {pair_title} · live training</h1>
<div class="sub">{cfg_html}</div>

{temp_banner}

{"<div class='done'>✓ training complete — " + str(metrics.get("total_minutes", "")) + " min. See " + self.out_dir + "/comparison.md</div>" if finished else ""}

<div class="cards">
  <div class="card temp"><div class="k">GPU temp</div><div class="v">{gpu_temp_str}</div><div class="k">{gpu_sub}</div></div>
  <div class="card"><div class="k">epoch</div><div class="v">{done} / {self.total_epochs}</div></div>
  <div class="card"><div class="k">elapsed</div><div class="v">{_fmt(elapsed)}</div></div>
  <div class="card"><div class="k">{"total time" if finished else "eta"}</div><div class="v">{_fmt(elapsed) if finished else eta_str}</div></div>
  <div class="card"><div class="k">val perplexity</div><div class="v">{last_ppl}</div></div>
  <div class="card"><div class="k">token accuracy</div><div class="v" style="color:#f59e0b">{last_acc}</div></div>
  <div class="card"><div class="k">best BLEU (test)</div><div class="v" style="color:#22c55e">{best_bleu_str}</div></div>
</div>
<div class="bar"><i></i></div>
<div class="status">{html.escape(self.status_line)}</div>

<p class="sub" style="margin:10px 0 0">GPU temperature guide —
<span style="color:#22c55e">&lt;{TEMP_OK}&deg;C safe</span> ·
<span style="color:#f59e0b">{TEMP_OK}–{TEMP_WARM}&deg;C warm</span> ·
<span style="color:#f97316">{TEMP_WARM}–{TEMP_HOT}&deg;C hot</span> ·
<span style="color:#ef4444">&ge;{TEMP_HOT}&deg;C danger, tell Claude to stop</span></p>

{bleu_section}

<h2>Curves</h2>
<div class="charts">
  <div>{loss_chart}</div>
  <div>{acc_chart}</div>
  <div>{ppl_chart}</div>
  <div>{kl_chart}</div>
  <div>{temp_chart}</div>
</div>

{dist_section}

<h2>Metrics</h2>
<table>
<tr><th>epoch</th><th>train loss</th><th>val loss (KL)</th><th>perplexity</th><th>accuracy</th><th>KL snap</th><th>BLEU val</th><th>BLEU test</th><th>time</th></tr>
{metric_rows}
</table>

{sample_html}

<p class="sub" style="margin-top:28px">auto-refreshes every {self.refresh_seconds}s · {time.strftime('%H:%M:%S')}</p>
</body></html>"""


def _fmt(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _svg_chart(series, width=380, height=220, pad=34, y_label=""):
    """Minimal multi-line SVG chart. ``series`` = [(label, [(x,y)...], color)]."""
    pts_all = [p for _, pts, _ in series for p in pts]
    if len(pts_all) < 2:
        return f"<svg width='{width}' height='{height}'></svg><div class='sub'>{y_label}: waiting for data…</div>"
    xs = [x for x, _ in pts_all]
    ys = [y for _, y in pts_all]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    y0 = min(y0, y1 - 1e-6)
    span_x = (x1 - x0) or 1
    span_y = (y1 - y0) or 1

    def sx(x):
        return pad + (x - x0) / span_x * (width - pad - 8)

    def sy(y):
        return height - pad - (y - y0) / span_y * (height - pad - 12)

    grid = ""
    for frac in (0, 0.5, 1):
        val = y0 + frac * span_y
        gy = sy(val)
        grid += f"<line x1='{pad}' y1='{gy:.1f}' x2='{width - 8}' y2='{gy:.1f}' stroke='currentColor' stroke-opacity='0.12'/>"
        grid += f"<text x='2' y='{gy + 3:.1f}' font-size='9' fill='currentColor' fill-opacity='0.5'>{val:.1f}</text>"

    lines = ""
    legend = ""
    for idx, (label, pts, color) in enumerate(series):
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        lines += f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'/>"
        last = pts[-1]
        lines += f"<circle cx='{sx(last[0]):.1f}' cy='{sy(last[1]):.1f}' r='3' fill='{color}'/>"
        legend += f"<tspan fill='{color}'>● </tspan><tspan>{html.escape(label)}  </tspan>"

    return (
        f"<svg width='{width}' height='{height}' style='color:var(--fg)'>"
        f"{grid}{lines}"
        f"<text x='{pad}' y='12' font-size='10' fill='currentColor'>{legend}</text>"
        f"</svg>"
    )


def _distribution_section(dist) -> str:
    """The 'model prediction vs. label-smoothed target' overlapping bar chart."""
    if not dist:
        return (
            "<h2>Prediction vs. target distribution</h2>"
            "<div class='sub'>computed after the first epoch — shows how the model's "
            "next-token probabilities line up with the smoothed training target.</div>"
        )
    labels = dist["labels"]
    model_p = dist["model"]
    target_p = dist["target"]
    ctx = html.escape(dist.get("context_token", ""))
    true_tok = html.escape(dist.get("true_token", ""))
    kl = dist.get("kl", 0.0)

    width, height, pad = 560, 240, 30
    n = len(labels)
    slot = (width - pad - 8) / max(n, 1)
    bw = slot * 0.36
    maxv = max(max(model_p, default=0.01), max(target_p, default=0.01), 0.01)

    def bar(i, v, color, offset):
        h = (v / maxv) * (height - pad - 22)
        x = pad + i * slot + offset
        y = height - pad - h
        return f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw:.1f}' height='{h:.1f}' fill='{color}' rx='2'/>"

    bars = ""
    for i, lab in enumerate(labels):
        bars += bar(i, model_p[i], "#4f8cff", 2)
        bars += bar(i, target_p[i], "#22c55e", 2 + bw + 2)
        lx = pad + i * slot + slot / 2
        mark = " ★" if lab == dist.get("true_token") else ""
        bars += (
            f"<text x='{lx:.1f}' y='{height - pad + 12:.1f}' font-size='9' "
            f"fill='currentColor' text-anchor='middle'>{html.escape(lab)}{mark}</text>"
        )

    svg = (
        f"<svg width='{width}' height='{height}' style='color:var(--fg)'>"
        f"<text x='{pad}' y='12' font-size='10' fill='currentColor'>"
        f"<tspan fill='#4f8cff'>■ </tspan>model  "
        f"<tspan fill='#22c55e'>■ </tspan>smoothed target</text>"
        f"{bars}</svg>"
    )
    return (
        "<h2>Prediction vs. target distribution</h2>"
        f"<div class='sub'>after context token <code>{ctx}</code>, the true next word is "
        f"<code>{true_tok}</code> (★). KL(target ‖ model) = <b>{kl:.3f}</b> — "
        f"the blue bars should climb toward the green as training proceeds.</div>"
        f"<div class='charts'><div>{svg}</div></div>"
    )
