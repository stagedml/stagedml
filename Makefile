.DEFAULT_GOAL = all
VERSION_STAGEDML = $(shell python3 setup.py --version)
WHEEL_STAGEDML_NAME = stagedml-$(VERSION_STAGEDML)-py3-none-any.whl
WHEEL_STAGEDML = ./docker/wheels/$(WHEEL_STAGEDML_NAME)

SRC_TFM = $(shell find ./3rdparty/tensorflow_models -name '*\.py')
VERSION_TFM = $(shell cat ./3rdparty/tensorflow_models/official/pip_package/setup.py | \
                      sed -n "s/.*version='\(.*\)'.*/\1/p")
WHEEL_TFM_NAME = tf_models_official-$(VERSION_TFM)-py3-none-any.whl
WHEEL_TFM = ./docker/wheels/$(WHEEL_TFM_NAME)

VERSION_TF = $(shell cat ./3rdparty/tensorflow/tensorflow/tensorflow.bzl | \
                     sed -n 's@.*VERSION = "\(.*\)".*@\1@p')
WHEEL_TF_NAME = tensorflow-$(VERSION_TF)-cp36-cp36m-linux_x86_64.whl
WHEEL_TF = ./docker/wheels/$(WHEEL_TF_NAME)

VERSION_PLYLIGHTNIX = $(shell cd ./3rdparty/pylightnix; python3 setup.py --version)
WHEEL_PYLIGHTNIX_NAME = /pylightnix-$(VERSION_PLYLIGHTNIX)-py3-none-any.whl
WHEEL_PYLIGHTNIX = ./docker/wheels/$(WHEEL_PYLIGHTNIX_NAME)

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

## Utils

.PHONY: check_nonroot, check_root
check_nonroot:
	test "$(shell whoami)" != "root"

check_root:
	test "$(shell whoami)" = "root"

## TensorFlow

$(WHEEL_TF): ./_tf/$(WHEEL_TF_NAME)
	-rm ./docker/wheels/tensorflow*
	cp ./_tf/$(WHEEL_TF_NAME) $(WHEEL_TF) || ( \
		echo "Build Tensorflow wheel manually by running \`buildtf\` shell command"; \
		exit 1 )

.PHONY: wheel_tf # We don't bother scanning for the changed sources here
wheel_tf: check_nonroot $(WHEEL_TF)

.PHONY: check_tf check_nonroot
check_tf: $(WHEEL_TF)
	pip3 hash $(WHEEL_TF) > .check_tf-stamp-$(HOSTNAME)
	diff .check_tf-stamp-$(HOSTNAME) .install_tf-stamp-$(HOSTNAME)

.PHONY: install_tf
install_tf: check_root # Has to be run by root
	test -f $(WHEEL_TF) || ( echo 'run `make wheel_tf` first'; exit 1; )
	pip3 install --force $(WHEEL_TF)
	pip3 hash $(WHEEL_TF) > .install_tf-stamp-$(HOSTNAME)
	pip3 install tensorflow_hub

## TensorFlow/Models

$(WHEEL_TFM): $(SRC_TFM)
	-rm ./docker/wheels/tf_models*
	( cd ./3rdparty/tensorflow_models && \
	  ( rm -rf build dist || true ) && \
	  python3 official/pip_package/setup.py sdist bdist_wheel \
	)
	cp ./3rdparty/tensorflow_models/dist/$(WHEEL_TFM_NAME) $(WHEEL_TFM)

.PHONY: wheel_tfm # We don't bother scanning for the changed sources here
wheel_tfm: check_nonroot $(WHEEL_TFM)

.PHONY: check_tfm check_nonroot
check_tfm: $(WHEEL_TFM)
	pip3 hash $(WHEEL_TFM) > .check_tfm-stamp-$(HOSTNAME)
	diff .check_tfm-stamp-$(HOSTNAME) .install_tfm-stamp-$(HOSTNAME)

.PHONY: install_tfm
install_tfm: check_root # Has to be run by root
	test -f $(WHEEL_TFM) || ( echo 'run `make wheel_tfm` first'; exit 1; )
	pip3 install --force $(WHEEL_TFM)
	pip3 hash $(WHEEL_TFM) > .install_tfm-stamp-$(HOSTNAME)


## Pylightnix

.PHONY: wheel_pylightnix
wheel_pylightnix: check_nonroot
	-rm ./docker/wheels/pylightnix*
	make -C 3rdparty/pylightnix wheel
	cp ./3rdparty/pylightnix/dist/$(WHEEL_PYLIGHTNIX_NAME) $(WHEEL_PYLIGHTNIX)

.PHONY: check_pylightnix
check_pylightnix: $(WHEEL_PYLIGHTNIX) check_nonroot
	pip3 hash $(WHEEL_PYLIGHTNIX) > .check_pylightnix-stamp-$(HOSTNAME)
	diff .check_pylightnix-stamp-$(HOSTNAME) .install_pylightnix-stamp-$(HOSTNAME)

.PHONY: install_pylightnix
install_pylightnix: check_root # Has to be run by root
	test -f $(WHEEL_PYLIGHTNIX) || ( echo 'run `make wheel_pylightnix` first'; exit 1; )
	pip3 install --force $(WHEEL_PYLIGHTNIX)
	pip3 hash $(WHEEL_PYLIGHTNIX) > .install_pylightnix-stamp-$(HOSTNAME)

## StagedML

$(WHEEL_STAGEDML): $(SRC) $(TESTS)
	-rm ./docker/wheels/stagedml*
	( cd $(STAGEDML_SOURCE) && \
		( rm -rf build dist || true ) && \
		python3 setup.py sdist bdist_wheel && \
		cp dist/$(WHEEL_STAGEDML_NAME) $(WHEEL_STAGEDML) \
	)

.PHONY: wheel_stagedml
wheel_stagedml: $(WHEEL_STAGEDML) check_nonroot

.PHONY: check_stagedml
check_stagedml: $(WHEEL_STAGEDML) check_nonroot
	pip3 hash $(WHEEL_STAGEDML) > .check_stagedml-stamp-$(HOSTNAME)
	diff .check_stagedml-stamp-$(HOSTNAME) .install_stagedml-stamp-$(HOSTNAME)

.PHONY: install_stagedml
install_stagedml: check_root # Has to be run by root
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

