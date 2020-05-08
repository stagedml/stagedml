.DEFAULT_GOAL = all

SRC = $(shell find -name '*\.py')

ifdef STAGEDML_SOURCE

SRC_STAGEDML = $(shell find $(STAGEDML_SOURCE)/src -name '*\.py')

.stamp_check: $(SRC_STAGEDML) $(STAGEDML_SOURCE)/Makefile Makefile
	make -C $(STAGEDML_SOURCE) check
	touch $@
else
.stamp_check:
	touch $@
endif

.stamp_trained: $(SRC) .stamp_check
	python3 $<
	touch $@

out/Report.md: Report.md.in Makefile $(SRC) .stamp_check .stamp_trained
	mkdir out; REPOUT=out REPIMG=. PYTHONPATH=. codebraid pandoc \
		-f markdown -t markdown --no-cache --overwrite --standalone \
		--self-contained -t gfm -o $@ $<
	@rm geckodriver.log || true

out_html/Report.html: Report.md.in Makefile $(SRC) .stamp_check .stamp_trained
	mkdir out_html ; REPOUT=`dirname $@` PYTHONPATH=. codebraid pandoc \
		-f markdown -t html --no-cache --overwrite --standalone --self-contained \
		--metadata pagetitle="RuSentiment" -o $@ $<
	@rm geckodriver.log || true

.PHONY: train
train: .stamp_trained

.PHONY: check
check: .stamp_check

.PHONY: html
html: out_html/Report.html

.PHONY: md
md: out/Report.md

.PHONY: all
all: md html
