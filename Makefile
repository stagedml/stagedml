.DEFAULT_GOAL = all
VERSION = $(shell cat setup.py | sed -n 's/.*version="\(.*\)".*/\1/p')
WHEEL = dist/stagedml-$(VERSION)-py3-none-any.whl
VERSION_TFM = $(shell cat 3rdparty/tensorflow_models/official/pip_package/setup.py | sed -n "s/.*version='\(.*\)'.*/\1/p")
WHEEL_TFM = 3rdparty/tensorflow_models/dist/tf_models_official-$(VERSION_TFM)-py3-none-any.whl
SRC = $(shell find src -name '*\.py')
TESTS = $(shell find tests -name '*\.py')

.stamp_tested: $(SRC) $(TESTS)
	pytest
	touch $@

.PHONY: test
test: .stamp_tested

## TensorFlow Models

.PHONY: wheels_tfm # We don't bother scanning for the changed sources here
wheels_tfm:
	( cd 3rdparty/tensorflow_models && \
	( rm -rf build dist || true ) && \
	python3 official/pip_package/setup.py sdist bdist_wheel )
	test -f $(WHEEL_TFM)
	@echo "\n\nTo install, run \`sudo -H make install_tfm\` or"
	@echo "> sudo -H pip3 install --force $(WHEEL_TFM)"

.PHONY: install_tfm
install_tfm: # Has to be run by root
	test -f $(WHEEL_TFM) || ( echo 'run `make wheels_tfm` first'; exit 1; )
	pip3 install --force $(WHEEL_TFM)
	pip3 hash $(WHEEL_TFM) > .install_tfm-stamp

.PHONY: check_tfm
check_tfm: $(WHEEL_TFM)
	pip3 hash $(WHEEL_TFM) > .check_tfm-stamp
	diff .check_tfm-stamp .install_tfm-stamp

## StagedML

$(WHEEL): $(SRC) $(TESTS)
	rm -rf build dist || true
	python3 setup.py sdist bdist_wheel
	test -f $@

.PHONY: check
check: $(WHEEL) check_tfm
	pip3 hash $(WHEEL) > .check-stamp
	make -C 3rdparty/pylightnix check
	diff .check-stamp .install-stamp

.PHONY: wheels
wheels: wheels_tfm $(WHEEL)
	make -C 3rdparty/pylightnix wheels
	@echo "\n\nTo install, run \`sudo -H make install\` or"
	@echo "> sudo -H pip3 install --force $(WHEEL)"

.PHONY: install
install: install_tfm
	test -f $(WHEEL) || ( echo 'run `make wheels` first'; exit 1; )
	make -C 3rdparty/pylightnix install
	pip3 install --force $(WHEEL)
	pip3 hash $(WHEEL) > .install-stamp

## All

.PHONY: all
all: test wheels
