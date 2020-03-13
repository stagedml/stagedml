.DEFAULT_GOAL = out/Report.html
SRC = $(shell find $(STAGEDML_ROOT)/src -name '*\.py')
TESTS = $(shell find $(STAGEDML_ROOT)/tests -name '*\.py')

.stamp_check: $(SRC) $(TESTS)
ifndef STAGEDML_ROOT
	$(error STAGEDML_ROOT env var should contain the path to the root of StagedML repo)
endif
	make -C $(STAGEDML_ROOT) check
	@if ! which pweave >/dev/null 2>&1 ; then \
		echo "pweave not found. Please install it with" ; \
		echo "> sudo -H pip3 install pweave" ; \
		exit 1 ; \
	fi
	touch $@

out/Report.md: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	PYTHONPATH=. pweave -i markdown -f markdown -o $@ $<
	! grep 'Traceback (most recent call last)' $@

out/Report.html: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	PYTHONPATH=. pweave --cache-results -i markdown -f md2html -o $@ $<
	! grep 'Traceback (most recent call last)' $@
