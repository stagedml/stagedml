.DEFAULT_GOAL = all
SRC = $(shell find $(STAGEDML_ROOT)/src -name '*\.py') \
			$(shell find $(STAGEDML_ROOT)/run/bert_finetune -name '*\.py')
TESTS = $(shell find $(STAGEDML_ROOT)/tests -name '*\.py')

.stamp_check: $(SRC) $(TESTS)
ifndef STAGEDML_ROOT
	$(error STAGEDML_ROOT env var should contain the path to the root of StagedML repo)
endif
	make -C $(STAGEDML_ROOT) check
	@if ! which codebraid >/dev/null 2>&1 ; then \
		echo "Please install `codebraid`" ; \
		exit 1 ; \
	fi
	touch $@

out/Report.md: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	mkdir out; REPOUT=out REPIMG=. PYTHONPATH=. codebraid pandoc \
		-f markdown -t markdown --no-cache --overwrite --standalone \
		--self-contained -t gfm  -o $@ $<
	-rm geckodriver.log

out_html/Report.html: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	mkdir out_html ; REPOUT=`dirname $@` PYTHONPATH=. codebraid pandoc \
		-f markdown -t html --no-cache --overwrite --standalone --self-contained \
		--metadata pagetitle="BERT Pretraining" -o $@ $<
	-rm geckodriver.log

.PHONY: html
html: out_html/Report.html

.PHONY: md
md: out/Report.md

.PHONY: all
all: md html
