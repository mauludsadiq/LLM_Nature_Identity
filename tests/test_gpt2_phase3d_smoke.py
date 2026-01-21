from pathlib import Path
import pytest
import subprocess
import sys


@pytest.mark.slow
def test_gpt2_phase3d_smoke():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    subprocess.check_call([sys.executable, "-m", "scripts.run_phase2d_gpt2"])
    subprocess.check_call([sys.executable, "-m", "scripts.run_phase3d_gpt2"])
    out_png = Path("out/gpt2_phase3d/identity_topography_3d.png")
    assert out_png.exists() and out_png.stat().st_size > 0
