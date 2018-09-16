import sys

with open(sys.argv[1]) as f:
  prev, pprev = '', ''
  for line in f:
    tkns = line.strip().split(' ')
    if len(tkns) < 2:
      print(line.strip())
    elif line.startswith('-DOCSTART-'):
      print(line.strip())
    else:
      current = tkns[1]
      if current == 'I-PER':
        if prev == 'PUNCT' and pprev == 'I-PER':
          pass
        elif prev == 'I-PER':
          pass
        else:
          tkns[1] = 'B-PER'

      if len(tkns) < 14:
        tkns += ['none'] * (14 - len(tkns))
      print(' '.join(tkns))

      pprev = prev
      prev = current
