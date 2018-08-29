import sys
from tokenizer import Tokenizer

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Wrong arguments')
    quit()

  tokenizer = Tokenizer()

  zero_padded = sys.argv[1].zfill(3)  
  html_file = 'htmls/' + zero_padded + '.html'
  target_file = 'target_names/target_names_' + zero_padded + '.txt'

  target_names = []
  with open(target_file) as f:
    for name in f:
      if len(name) == 0: 
        continue
      name = " ".join(tokenizer.tokenize_text(name.strip()))
      target_names.append(name.strip())

  names = [] 
  with open(html_file) as f:
    sentences = tokenizer.tokenize(f.read(), target_names)
 
    name = [] 
    for i, s in enumerate(sentences):
      for t in s:
        if t.bio_tag == 'I-PER':
          name.append(t.tkn)
        elif t.bio_tag == 'B-PER':
          if len(name) > 0:
            names.append(' '.join(name))
            name = []
          name.append(t.tkn)
        elif len(name) > 0:
          names.append(' '.join(name))
          name = []

  for n in names:
    if not n in target_names:
      print('Oh noes', n)
      quit()

  for n in target_names:
    if not n in names:
      print('Oh noes', n)
      quit()

  print('Oh yeah')
