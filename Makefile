.DEFAULT_GOAL = all
VERSION_STAGEDML = $(shell cat $(STAGEDML_ROOT)/setup.py | \
                  sed -n 's/.*version="\(.*\)".*/\1/p')
WHEEL_STAGEDML_NAME = stagedml-$(VERSION_STAGEDML)-py3-none-any.whl
WHEEL_STAGEDML = $(STAGEDML_ROOT)/docker/wheels/$(WHEEL_STAGEDML_NAME)

VERSION_TFM = $(shell cat $(STAGEDML_ROOT)/3rdparty/tensorflow_models/official/pip_package/setup.py | \
                      sed -n "s/.*version='\(.*\)'.*/\1/p")
WHEEL_TFM_NAME = tf_models_official-$(VERSION_TFM)-py3-none-any.whl
WHEEL_TFM = $(STAGEDML_ROOT)/docker/wheels/$(WHEEL_TFM_NAME)

VERSION_TF = $(shell cat $(STAGEDML_ROOT)/3rdparty/tensorflow/tensorflow/tensorflow.bzl | \
                     sed -n 's@.*VERSION = "\(.*\)".*@\1@p')
WHEEL_TF_NAME = tensorflow-$(VERSION_TF)-cp36-cp36m-linux_x86_64.whl
WHEEL_TF = $(STAGEDML_ROOT)/docker/wheels/$(WHEEL_TF_NAME)

VERSION_PLYLIGHTNIX = $(shell cat $(STAGEDML_ROOT)/3rdparty/pylightnix/setup.py | \
                              sed -n 's/.*version="\(.*\)".*/\1/p')
WHEEL_PYLIGHTNIX_NAME = /pylightnix-$(VERSION_PLYLIGHTNIX)-py3-none-any.whl
WHEEL_PYLIGHTNIX = $(STAGEDML_ROOT)/docker/wheels/$(WHEEL_PYLIGHTNIX_NAME)

SRC = $(shell find src -name '*\.py')
TESTS = $(shell find tests -name '*\.py')

.stamp_tested: $(SRC) $(TESTS)
	pytest
	touch $@

.PHONY: test
test: .stamp_tested

.PHONY: typecheck tc
typecheck:
	pytest --mypy -m mypy
tc: typecheck

## TensorFlow

.PHONY: wheel_tf # We don't bother scanning for the changed sources here
wheel_tf:
	-rm $(STAGEDML_ROOT)/docker/wheels/tensorflow*
	cp $(STAGEDML_ROOT)/_tf/$(WHEEL_TF_NAME) $(WHEEL_TF) || ( \
		echo "Build Tensorflow wheel manually by running `buildtf` shell command"; \
		exit 1 )

.PHONY: check_tf
check_tf: $(WHEEL_TF)
	pip3 hash $(WHEEL_TF) > .check_tf-stamp-$(HOSTNAME)
	diff .check_tf-stamp-$(HOSTNAME) .install_tf-stamp-$(HOSTNAME)

.PHONY: install_tf
install_tf: # Has to be run by root
	test "$(shell whoami)" = "root"
	test -f $(WHEEL_TF) || ( echo 'run `make wheel_tf` first'; exit 1; )
	pip3 install --force $(WHEEL_TF)
	pip3 hash $(WHEEL_TF) > .install_tf-stamp-$(HOSTNAME)

## TensorFlow Models

.PHONY: wheel_tfm # We don't bother scanning for the changed sources here
wheel_tfm:
	-rm $(STAGEDML_ROOT)/docker/wheels/tf_models*
	( cd $(STAGEDML_ROOT)/3rdparty/tensorflow_models && \
	  ( rm -rf build dist || true ) && \
	  python3 official/pip_package/setup.py sdist bdist_wheel && \
		cp dist/$(WHEEL_TFM_NAME) $(WHEEL_TFM) \
	)

.PHONY: check_tfm
check_tfm: $(WHEEL_TFM)
	pip3 hash $(WHEEL_TFM) > .check_tfm-stamp-$(HOSTNAME)
	diff .check_tfm-stamp-$(HOSTNAME) .install_tfm-stamp-$(HOSTNAME)

.PHONY: install_tfm
install_tfm: # Has to be run by root
	test "$(shell whoami)" = "root"
	test -f $(WHEEL_TFM) || ( echo 'run `make wheel_tfm` first'; exit 1; )
	pip3 install --force $(WHEEL_TFM)
	pip3 hash $(WHEEL_TFM) > .install_tfm-stamp-$(HOSTNAME)


## Pylightnix

.PHONY: wheel_pylightnix
wheel_pylightnix:
	-rm $(STAGEDML_ROOT)/docker/wheels/pylightnix*
	make -C 3rdparty/pylightnix wheel
	cp $(STAGEDML_ROOT)/3rdparty/pylightnix/dist/$(WHEEL_PYLIGHTNIX_NAME) $(WHEEL_PYLIGHTNIX)

.PHONY: check_pylightnix
check_pylightnix: $(WHEEL_PYLIGHTNIX)
	pip3 hash $(WHEEL_PYLIGHTNIX) > .check_pylightnix-stamp-$(HOSTNAME)
	diff .check_pylightnix-stamp-$(HOSTNAME) .install_pylightnix-stamp-$(HOSTNAME)

.PHONY: install_pylightnix
install_pylightnix: # Has to be run by root
	test "$(shell whoami)" = "root"
	test -f $(WHEEL_PYLIGHTNIX) || ( echo 'run `make wheel_pylightnix` first'; exit 1; )
	pip3 install --force $(WHEEL_PYLIGHTNIX)
	pip3 hash $(WHEEL_PYLIGHTNIX) > .install_pylightnix-stamp-$(HOSTNAME)

## StagedML

$(WHEEL_STAGEDML): $(SRC) $(TESTS)
	-rm $(STAGEDML_ROOT)/docker/wheels/stagedml*
	( cd $(STAGEDML_ROOT) && \
		( rm -rf build dist || true ) && \
		python3 setup.py sdist bdist_wheel && \
		cp dist/$(WHEEL_STAGEDML_NAME) $(WHEEL_STAGEDML) \
	)

.PHONY: wheel_stagedml
wheel_stagedml: $(WHEEL_STAGEDML)

.PHONY: check_stagedml
check_stagedml: $(WHEEL_STAGEDML)
	pip3 hash $(WHEEL_STAGEDML) > .check_stagedml-stamp-$(HOSTNAME)
	diff .check_stagedml-stamp-$(HOSTNAME) .install_stagedml-stamp-$(HOSTNAME)

.PHONY: install_stagedml
install_stagedml: # Has to be run by root
	test "$(shell whoami)" = "root"
	test -f $(WHEEL_STAGEDML) || ( echo 'run `make wheel` first'; exit 1; )
	pip3 install --force $(WHEEL_STAGEDML)
	pip3 hash $(WHEEL_STAGEDML) > .install_stagedml-stamp-$(HOSTNAME)

## All

.PHONY: wheel check install
wheel: wheel_tf wheel_tfm wheel_pylightnix wheel_stagedml
	@echo "\n\nTo install wheels, run \`sudo -E make install\`"
check: check_tf check_tfm check_pylightnix check_stagedml
install:            install_tfm install_pylightnix install_stagedml
	#      ^^^^^^^^^
	# We don't install TF automatically to save time

.PHONY: all
all: test wheel

