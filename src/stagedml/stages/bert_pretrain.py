from pylightnix import ( Manager, Lens, DRef, Build, RefPath, mklens, mkdrv,
    build_wrapper, build_path, mkconfig, match_only, promise, build_outpath )
from stagedml.imports import ( walk, abspath, join, Random, partial, cpu_count,
    getpid, makedirs)
from stagedml.utils import ( concat, batch )
from official.nlp.bert.tokenization import FullTokenizer, convert_to_unicode
from official.nlp.bert.create_pretraining_data import (
    create_instances_from_document, write_instance_to_example_files, TrainingInstance )

from stagedml.types import ( List, Optional, Wikidump )

from bz2 import ( open as bz2_open )
from json import ( loads as json_loads )

from multiprocessing.pool import Pool

TokSentence=List[int]
TokDoc=List[TokSentence]

def tokenize_file(input_file:str, tokenizer:FullTokenizer)->List[TokDoc]:
  chars=[chr(ord('a')+x) for x in range(ord('z')-ord('a'))]
  print(chars[getpid()%len(chars)], end='', flush=True)
  all_documents:List[TokDoc] = [[]]
  sentences_skipped=0
  sentences_tokenized=0
  with bz2_open(input_file, "rt", encoding='utf-8') as reader:
    for line in reader:
      d=json_loads(line)
      sentences=d['text'].split('\n')

      for i,s in enumerate(sentences):
        s=s.strip()
        tokens:Optional[TokSentence] = tokenizer.tokenize(s)
        if tokens:
          all_documents[-1].append(tokens)
          sentences_tokenized+=1
        else:
          if len(s)>0:
            sentences_skipped+=1
          else:
            sentences_tokenized+=1
      all_documents.append([])
  all_documents = [x for x in all_documents if len(x)>0]
  return all_documents



def realize_pretraining(b:Build)->None:
  tokenizer = FullTokenizer(vocab_file=mklens(b).vocab_file.syspath,
                            do_lower_case=mklens(b).do_lower_case.val)

  input_files = []
  for root, dirs, filenames in walk(mklens(b).input_folder.syspath, topdown=True):
    for filename in sorted(filenames):
      if filename.endswith('bz2'):
        input_files.append(abspath(join(root, filename)))

  max_seq_length=mklens(b).max_seq_length.val
  dupe_factor=mklens(b).dupe_factor.val
  short_seq_prob=mklens(b).short_seq_prob.val
  masked_lm_prob=mklens(b).masked_lm_prob.val
  max_predictions_per_seq=mklens(b).max_predictions_per_seq.val
  do_whole_word_mask=mklens(b).do_whole_word_mask.val
  rng=Random(mklens(b).random_seed.val)
  ncpu=round((cpu_count() or 1)*0.75)
  outdir=mklens(b).output.syspath
  makedirs(outdir)

  for ibatch,ifiles in enumerate(batch(input_files,mklens(b).input_files_chunk.val)):
    print(f"Reading next {len(ifiles)} files")
    ofiles=[join(outdir,'output-%02d-%02d.tfrecord'%(ibatch,nout)) for nout in \
            range(mklens(b).output_files_per_chunk.val)]

    with Pool(processes=ncpu) as p:
      all_documents=\
          concat(p.map(
            partial(tokenize_file,tokenizer=tokenizer),
            ifiles, chunksize=max(1,len(ifiles)//(4*ncpu))))

    print(f'Read {len(all_documents)} documents')
    print('Shuffling documents')
    rng.shuffle(all_documents)

    print('Producing instances')
    vocab_words = list(tokenizer.vocab.keys())
    instances:List[TrainingInstance] = []
    for _ in range(dupe_factor):
      for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents, document_index, max_seq_length, short_seq_prob,
                masked_lm_prob, max_predictions_per_seq, vocab_words, rng,
                do_whole_word_mask))

    print(f'Produced {len(instances)} instances')
    print('Shuffling instances')
    rng.shuffle(instances)

    print('Writing tfrecords')
    write_instance_to_example_files(
      instances, tokenizer,
      mklens(b).max_seq_length.val,
      mklens(b).max_predictions_per_seq.val,
      ofiles,
      gzip_compress=True)
    print('')


def bert_pretraining_tfrecords(m:Manager, vocab_file:RefPath, wiki:DRef)->DRef:

  def _config():
    name='bert_pretraining'
    max_seq_length=128
    max_predictions_per_seq=20
    random_seed=17
    dupe_factor=10
    short_seq_prob=0.1
    masked_lm_prob=0.15
    do_lower_case=True
    do_whole_word_mask=False
    nonlocal vocab_file
    input_folder=mklens(wiki).output.refpath
    input_files_chunk=100
    output_files_per_chunk=4
    output=[promise,'output']
    return mkconfig(locals())

  return mkdrv(m,
    config=_config(),
    matcher=match_only(),
    realizer=build_wrapper(realize_pretraining))



