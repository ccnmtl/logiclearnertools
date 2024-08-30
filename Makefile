PY_DIRS=logictools
VE ?= ./ve
REQUIREMENTS ?= requirements.txt
SYS_PYTHON ?= python3
PIP ?= $(VE)/bin/pip
PY_SENTINAL ?= $(VE)/sentinal
WHEEL_VERSION ?= 0.44.0
PIP_VERSION ?= 24.2
MAX_COMPLEXITY ?= 7
FLAKE8 ?= $(VE)/bin/flake8
PIP ?= $(VE)/bin/pip

$(PY_SENTINAL):
	rm -rf $(VE)
	$(SYS_PYTHON) -m venv $(VE)
	$(PIP) install pip==$(PIP_VERSION)
	$(PIP) install --upgrade setuptools
	$(PIP) install wheel==$(WHEEL_VERSION)
	$(PIP) install --no-deps --requirement $(REQUIREMENTS)
	touch $@

all: flake8 test

clean:
	rm -rf $(VE)
	find . -name '*.pyc' -exec rm {} \;

test: $(PY_SENTINAL)
	$(PY_DIRS)

flake8: $(PY_SENTINAL)
	$(FLAKE8) $(PY_DIRS)
