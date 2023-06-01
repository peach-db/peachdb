setup:
	pip install -e .
	pre-commit install

release:
	bumpversion patch --allow-dirty
