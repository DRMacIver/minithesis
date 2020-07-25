venv/bin/python:
	virtualenv venv

venv/installed: venv/bin/python requirements.txt
	venv/bin/pip install -r requirements.txt
	touch venv/installed

.PHONY: update-requirements
update-requirements: venv/installed
	venv/bin/pip freeze > requirements.txt
	

.PHONY: test
test: venv/installed
	venv/bin/python -m coverage run --include=minithesis.py --branch -m pytest test_minithesis.py --ff --maxfail=1 -m 'not hypothesis' --durations=100 --verbose
	venv/bin/coverage report --show-missing --fail-under=100

.PHONY: format
format: venv/installed
	venv/bin/isort *.py
	venv/bin/black *.py
