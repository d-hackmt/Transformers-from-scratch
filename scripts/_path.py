"""Make the project root importable when a script is run directly.

``python scripts/foo.py`` only puts ``scripts/`` on ``sys.path``, not the repo
root, so ``import annotated_transformer`` would fail.  Every script does
``import _path  # noqa`` first to fix that.  (Not needed if you install the
package or run ``python -m``.)
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
