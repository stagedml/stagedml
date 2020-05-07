.DEFAULT_GOAL = all
SRC = $(shell find -name '*\.py')

ifdef STAGEDML_ROOT
SRC_STAGEDML = $(shell find $(STAGEDML_ROOT)/src -name '*\.py')
.stamp_check: $(SRC_STAGEDML) $(STAGEDML_ROOT)/Makefile Makefile
	make -C $(STAGEDML_ROOT) check
	touch $@
else
.stamp_check:
	touch $@
endif

.stamp_trained: .stamp_check $(SRC)
	python3 ./bert_finetune_experiment.py
	touch $@

out/Report.md: Report.md.in Makefile $(SRC) .stamp_trained
	mkdir out; REPOUT=out REPIMG=. PYTHONPATH=. codebraid pandoc \
		-f markdown -t markdown --no-cache --overwrite --standalone \
		--self-contained -t gfm  -o $@ $<
	-rm geckodriver.log

out_html/Report.html: Report.md.in Makefile $(SRC) .stamp_trained
	mkdir out_html ; REPOUT=`dirname $@` PYTHONPATH=. codebraid pandoc \
		-f markdown -t html --no-cache --overwrite --standalone --self-contained \
		--metadata pagetitle="BERT finetuning" -o $@ $<
	-rm geckodriver.log

.PHONY: train
train: .stamp_trained

.PHONY: html
html: out_html/Report.html

.PHONY: md
md: out/Report.md

.PHONY: all
all: md html
