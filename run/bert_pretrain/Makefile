.DEFAULT_GOAL = all
SRC = $(shell find $(STAGEDML_ROOT)/src -name '*\.py') \
			$(STAGEDML_ROOT)/run/bert_pretrain/main.py
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

Report.md: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	PYTHONPATH=. codebraid pandoc -f markdown -t markdown --no-cache \
						             --overwrite --standalone --self-contained \
												 -o $@ $<

Report.html: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
	PYTHONPATH=. codebraid pandoc -f markdown -t html --no-cache \
						             --overwrite --standalone --self-contained \
												 -o $@ $<

# out/Report.py: Report.md.in Makefile $(SRC) $(TESTS) .stamp_check
# 	ptangle -i markdown $<
# 	mv Report.md.py $@

.PHONY: all
all: Report.md Report.html
