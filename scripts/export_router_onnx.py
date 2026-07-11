"""Export the trained router to ONNX (wrapper for ``arcs.router.export_onnx``)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arcs.router.export_onnx import main

if __name__ == "__main__":
    main()
