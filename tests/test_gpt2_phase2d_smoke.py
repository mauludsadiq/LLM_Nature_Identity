from pathlib import Path
import pytest
import subprocess
import sys


@pytest.mark.slow
def test_gpt2_phase2d_smoke():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    subprocess.check_call([sys.executable, "-m", "scripts.run_phase2d_gpt2"])
    out_json = Path("out/gpt2_phase2d/phase_transition_2d.json")
    out_png = Path("out/gpt2_phase2d/phase_transition_2d.png")
    assert out_json.exists() and out_json.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
