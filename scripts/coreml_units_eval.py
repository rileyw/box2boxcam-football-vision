"""Fork-only: run evaluate_detector.py with Core ML forced onto chosen compute units.

Ultralytics loads every Core ML model on CPU_AND_NE (falling back to CPU_ONLY), with
no option to choose. This wrapper overrides the ``compute_units`` that
``coremltools.models.MLModel`` receives, then runs the evaluation script unchanged::

    COREML_UNITS=CPU_ONLY python scripts/coreml_units_eval.py --weights m.mlpackage ...

COREML_UNITS is a ``coremltools.ComputeUnit`` name: CPU_ONLY, CPU_AND_NE, CPU_AND_GPU, ALL.
"""

import os
import runpy
import sys
from pathlib import Path

import coremltools as ct

units = getattr(ct.ComputeUnit, os.environ.get("COREML_UNITS", "CPU_AND_NE"))
original_init = ct.models.MLModel.__init__


def forced_init(self, *args, **kwargs):
    kwargs["compute_units"] = units
    original_init(self, *args, **kwargs)


ct.models.MLModel.__init__ = forced_init
print(f"Core ML compute units forced to {units.name}", file=sys.stderr)

script = Path(__file__).with_name("evaluate_detector.py")
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
