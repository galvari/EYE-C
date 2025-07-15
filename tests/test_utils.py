import sys
import types
from pathlib import Path

# Provide a minimal numpy stub so that src.utils can be imported without numpy
if 'numpy' not in sys.modules:
    sys.modules['numpy'] = types.ModuleType('numpy')

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from utils import compute_iou


def test_iou_full_intersection():
    bb1 = [0, 0, 10, 10]
    bb2 = [0, 0, 10, 10]
    assert compute_iou(bb1, bb2) == 1.0


def test_iou_no_intersection():
    bb1 = [0, 0, 10, 10]
    bb2 = [20, 20, 30, 30]
    assert compute_iou(bb1, bb2) == 0.0


def test_iou_partial_intersection():
    bb1 = [0, 0, 10, 10]
    bb2 = [5, 5, 15, 15]
    expected = 25 / 175  # 0.14285714285714285
    assert abs(compute_iou(bb1, bb2) - expected) < 1e-6