.DEFAULT_GOAL = all
SRC = $(shell find $(STAGEDML_SOURCE)/src -name '*\.py') $(STAGEDML_SOURCE)/run/nl2bash/analyze.py
TESTS = $(shell find $(STAGEDML_SOURCE)/tests -name '*\.py')

.stamp_check: $(SRC) $(TESTS)
ifndef STAGEDML_SOURCE
	$(error STAGEDML_SOURCE env var should contain the path to the root of StagedML repo)
endif
	make -C $(STAGEDML_SOURCE) check
	@if ! which pweave >/dev/null 2>&1 ; then \
		echo "pweave not found. Please install it with" ; \
		echo "> sudo -H pip3 install pweave" ; \
		exit 1 ; \
	fi
	touch $@

out/Report.md: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	-rm `dirname $@`/figures/*png
	REPORT_OUTPATH=`dirname $@` PYTHONPATH=. pweave -i markdown -f markdown -o $@.tmp $<
	! grep 'Traceback (most recent call last)' $@.tmp
	mv $@.tmp $@

out_html/Report.html: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	-rm `dirname $@`/figures/*png
	REPORT_OUTPATH=`dirname $@` PYTHONPATH=. pweave -i markdown -f md2html -o $@ $<
	! grep 'Traceback (most recent call last)' $@

out/Report.py: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	ptangle -i markdown $<
	mv Report.md.py $@

.PHONY: all
all: out/Report.md out/Report.py
