.PHONY: install syntax test validate smoke train clean-artifacts

install:
	pip install --upgrade pip
	pip install -r requirements.txt

syntax:
	python -m py_compile train_milk10k_transformer.py scripts/*.py tests/*.py

test:
	pytest -q

validate:
	python scripts/validate_csvs.py --root .

smoke:
	python scripts/run_tiny_debug.py

train:
	python train_milk10k_transformer.py

clean-artifacts:
	find checkpoints logs outputs -type f ! -name 'README.md' ! -name '.gitkeep' -delete
