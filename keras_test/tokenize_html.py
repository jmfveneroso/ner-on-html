#!/usr/bin/python
# coding=UTF-8

import sys
from tokenizer import Tokenizer

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Wrong arguments')
    quit()

  tokenizer = Tokenizer()

  with open(sys.argv[1]) as f:
    sentences = tokenizer.tokenize(f.read())
    for s in sentences:
      for t in s:
        sys.stdout.write(t.tkn + ' ')
