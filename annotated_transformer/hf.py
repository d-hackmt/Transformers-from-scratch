"""Resolve checkpoint / vocab paths, falling back to a Hugging Face repo.

The trained checkpoints and the vocab cache are large, so they live in a Hugging
Face model repo rather than in git.  :func:`resolve` lets the rest of the code
keep asking for a normal local path - if the file is there it is used as-is; if
not, it is downloaded from the HF repo and cached (in ``~/.cache/huggingface``).

So you can ``rm results*/checkpoints/*.pt`` and ``vocab.pt`` locally and every
script still works - it just pulls from HF the first time.

Repo: https://huggingface.co/Diveshj/transformer-from-scratch  (public)

Path handling
-------------
* an existing local path                     -> returned unchanged
* ``"de-en:10"`` / ``"en-de:15"`` shorthand   -> the matching checkpoint
* anything else                              -> looked up in the HF repo, where
  the local ``results/`` folder is named ``results_deen/``
"""

import os
import re

HF_REPO = "Diveshj/transformer-from-scratch"

# local top-level folder  ->  folder name inside the HF repo
# (the DE->EN run has been called both "results" and "results_de_en" locally;
#  on the HF repo that folder is "results_deen")
_DIRMAP = {"results": "results_deen", "results_de_en": "results_deen"}


def _expand_shorthand(spec: str):
    """``de-en:10`` -> ``results_de_en/checkpoints/multi30k_de_en_10ep.pt`` (else None)."""
    m = re.fullmatch(r"(de-en|en-de):(\d+)", spec.strip())
    if not m:
        return None
    direction, epoch = m.groups()
    tag = direction.replace("-", "_")
    folder = "results_de_en" if direction == "de-en" else "results_ende"
    return f"{folder}/checkpoints/multi30k_{tag}_{epoch}ep.pt"


def _to_repo_path(local_relpath: str) -> str:
    parts = local_relpath.replace("\\", "/").split("/")
    parts[0] = _DIRMAP.get(parts[0], parts[0])
    return "/".join(parts)


def resolve(path: str) -> str:
    """Return a usable local path for ``path`` (see module docstring)."""
    expanded = _expand_shorthand(path)
    if expanded is not None:
        path = expanded

    if os.path.exists(path):
        return path

    from huggingface_hub import hf_hub_download

    repo_file = _to_repo_path(path)
    print(f"fetching {repo_file} from {HF_REPO} ...", flush=True)
    return hf_hub_download(repo_id=HF_REPO, filename=repo_file, repo_type="model")


def direction_of(spec: str):
    """If ``spec`` is a shorthand or a known checkpoint name, return 'de-en'/'en-de'."""
    m = re.match(r"(de-en|en-de):", spec.strip())
    if m:
        return m.group(1)
    name = os.path.basename(spec)
    if "en_de" in name:
        return "en-de"
    if "de_en" in name:
        return "de-en"
    return None
