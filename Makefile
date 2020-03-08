.DEFAULT_GOAL = all
VERSION = $(shell cat setup.py | sed -n 's/.*version="\(.*\)".*/\1/p')
WHEEL = dist/stagedml-$(VERSION)-py3-none-any.whl
SRC = $(shell find src -name '*\.py')
TESTS = $(shell find tests -name '*\.py')

.stamp_tested: $(SRC) $(TESTS)
	pytest
	touch $@

.PHONY: test
test: .stamp_tested

$(WHEEL): $(SRC) $(TESTS)
	rm -rf build dist || true
	python3 setup.py sdist bdist_wheel
	test -f $@

.PHONY: wheels
wheels: $(WHEEL)
	@echo "To install, run \`sudo -H make install\` or"
	@echo "> sudo -H pip3 install --force $(WHEEL)"

.PHONY: install
install: $(WHEEL)
	make -C 3rdparty/pylightnix install
	pip3 install --force $(WHEEL)

.PHONY: all
all: test wheels
