from collections import defaultdict
from typing import Dict, List
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

def text2png(text, fullpath,
    color="#000", bgcolor="#FFF",
    fontfullpath=None, fontsize=13, leftpadding=3, rightpadding=3, width=200):
  """ Text2image
  Ref. https://gist.github.com/destan/5540702
  """
  REPLACEMENT_CHARACTER = u'\uFFFD'
  NEWLINE_REPLACEMENT_STRING = ' ' + REPLACEMENT_CHARACTER + ' '

  font = ImageFont.load_default() if fontfullpath is None else ImageFont.truetype(fontfullpath, fontsize)
  text = text.replace('\n', NEWLINE_REPLACEMENT_STRING)

  lines = []
  line = u""

  for word in text.split():
    if word == REPLACEMENT_CHARACTER: #give a blank line
      lines.append( line[1:] ) #slice the white space in the begining of the line
      line = u""
      lines.append( u"" ) #the blank line
    elif font.getsize( line + ' ' + word )[0] <= (width - rightpadding - leftpadding):
      line += ' ' + word
    else: #start a new line
      lines.append( line[1:] ) #slice the white space in the begining of the line
      line = u""
      #TODO: handle too long words at this point
      line += ' ' + word #for now, assume no word alone can exceed the line width

  if len(line) != 0:
    lines.append( line[1:] ) #add the last line

  img_height = sum([font.getsize(line)[1] for line in lines])
  img_width = leftpadding + max([font.getsize(line)[0] for line in lines]) + rightpadding
  img = Image.new("RGBA", (img_width, img_height), bgcolor)
  draw = ImageDraw.Draw(img)

  y = 0
  for line in lines:
    draw.text( (leftpadding, y), line, color, font=font)
    y += font.getsize(line)[1]

  img.save(fullpath)




def test_text2png():
  text2png(("This is adasdas asdashdasd asdajshdalsja asdaksjhasdas asldkasjdhlasd "
      "asldhasdlashdl alsdhasldjkashldkas aslkdjahskldashdlasjd asldjkashdlkajsdha "
      "asldkjash\nasdas фывфывф \n\n asdasd"),
      fullpath='test.png',
      fontsize=18,
      width=600,
      bgcolor='#262626',
      fontfullpath='font.ttf')

from typing import List
from pylightnix import ( mklens, instantiate, realize, PYLIGHTNIX_TMP, join,
    basename, rref2path, RRef )
from stagedml.stages.all import all_wmtsubtok_enru, all_transformer_wmtenru


OUT_DIR=rref2path(RRef('rref:f3ff5d54e4742005c8a9e234ff0de185-17a6e92a17d834f5715c8e3df70abda7-transformer_wmt'))
OUT_PATTERN=join(OUT_DIR,'output-{epoch}.txt')



def make_gifcards(nepoches:int=52,
                  samples:List[int]=[3,8,88,100],
                  outpat:str=OUT_PATTERN):
  rref=realize(instantiate(all_wmtsubtok_enru))
  print(rref)
  inp_txt=mklens(rref).eval_input_combined.syspath
  tgt_txt=mklens(rref).eval_target_combined.syspath
  print(inp_txt, tgt_txt)

  def _iter(out_txt:str):
    sindices=set(samples)
    with open(inp_txt,'r') as finp,\
         open(tgt_txt,'r') as ftgt,\
         open(out_txt,'r') as fout:
      for (idx,(inp,tgt,out)) in enumerate(zip(finp,ftgt,fout)):
        if idx in sindices:
          yield idx,inp,tgt,out

  for epoch in range(nepoches):
    out_txt=outpat.format(epoch=('?' if epoch==0 else str(epoch)))
    for (idx,inp,tgt,out) in _iter(out_txt):
      print(f'========== {basename(out_txt)}/{idx} ===========')
      print(inp)
      print(tgt)
      print(out)
      cardname=f'card-{idx:04d}-{epoch:03d}.png'
      print(f'Saving {cardname}')
      text2png(f"{inp}{tgt}Epoch {epoch}:\n{out}",
          fullpath=cardname,
          fontsize=18,
          width=600,
          bgcolor='#262626',
          color='#DCDAD5',
          fontfullpath='font.ttf')

def make_trlistings(nepoches:int=52,
                    samples:List[int]=[3,8,88,100,333,480,1405],
                    outpat:str=OUT_PATTERN):
  rref=realize(instantiate(all_wmtsubtok_enru))
  print(rref)
  inp_txt=mklens(rref).eval_input_combined.syspath
  tgt_txt=mklens(rref).eval_target_combined.syspath
  print(inp_txt, tgt_txt)

  def _iter(out_txt:str):
    sindices=set(samples)
    with open(inp_txt,'r') as finp,\
         open(tgt_txt,'r') as ftgt,\
         open(out_txt,'r') as fout:
      for (idx,(inp,tgt,out)) in enumerate(zip(finp,ftgt,fout)):
        if idx in sindices:
          yield idx,inp,tgt,out

  inps={}; outs:Dict[int,list]=defaultdict(list)
  for epoch in range(nepoches):
    out_txt=outpat.format(epoch=('?' if epoch==0 else str(epoch)))
    for (idx,inp,tgt,out) in _iter(out_txt):
      inps[idx]=(inp.strip(),tgt.strip())
      outs[idx].append(out.strip())

  for (idx,(inp,tgt)) in inps.items():
    cardname=f'trlisting-{idx:04d}.png'
    print(f'Saving {cardname}')
    eol='\n'
    text2png(f"Вход: {inp}\nЭталон: {tgt}\nvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n{eol.join(outs[idx])}",
        fullpath=cardname,
        fontsize=13,
        width=2*600,
        bgcolor='#262626',
        color='#DCDAD5',
        fontfullpath='font.ttf')

