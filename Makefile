.PHONY: venv test demo lint clean

venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -r requirements.txt

test:
	. .venv/bin/activate && pytest -q

demo:
	. .venv/bin/activate && python -m llm_identity.cli --out-dir out --seed 0 --preset conscious_agent

clean:
	rm -rf .venv __pycache__ .pytest_cache out
