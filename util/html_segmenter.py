#!/usr/bin/python
# coding=UTF-8

import sys
import re
from math import log
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

def remove_accents(tkn):
  text = tkn.strip().lower()

  special_chars = "ÀÁÂÃÄÅÆÇÈÉÊËÌÍİÎÏÐÑÒÓÔÕÖĞ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿšŽčžŠšČłńężśćŞ"
  chars         = "aaaaaaeceeeeiiiiidnooooogxouuuuypsaaaaaaeceeeeiiiionooooooouuuuypyszczssclnezscs"

  new_text = ""
  for c in text:
    index = special_chars.find(c)
    if index != -1:
      new_text += chars[index]
    else:
      new_text += c

  return new_text 

def is_punctuation(text):
  return re.match("^[\,\;\:\-\"\(\)“”；]$", text)

def tokenize_text(text):
  tkns = re.compile("(\s+|[\,\;\:\-\"\(\)“”；])").split(text)
  return [t for t in tkns if re.compile("\s+").match(t) is None and len(t) > 0] 

class HtmlToken():
  def __init__(self, tkn, element):
    self.tkn = tkn
    self.element = element
    self.features = []
    self.bio_tag = 'O'
    self.set_features()

  def is_punctuation(self):
    return 1 if re.match("^[\,\;\:\-\"\(\)“”；]$", self.tkn) else 0

  def is_email(self):
    return 1 if re.match('\S+@\S+(\.\S+)+', self.tkn) else 0

  def is_url(self):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return 1 if re.match(regex, self.tkn) else 0

  def is_number(self):
    return 1 if any(c.isdigit() for c in self.tkn) else 0

  def is_capitalized(self):
    if len(self.tkn) > 0:
      return 1 if self.tkn[0].isupper() else 0
    return 0

  def is_title(self):
    titles = ['m\.sc\.','sc\.nat\.','rer\.nat\.','sc\.nat\.','md\.',
      'b\.sc\.', 'bs\.sc\.', 'ph\.d\.', 'ed\.d\.', 'm\.s\.', 'hon\.', 
      'a\.d\.', 'em\.', 'apl\.', 'prof\.', 'prof\.dr\.', 'conf\.dr\.',
      'asist\.dr\.', 'dr\.', 'mr\.', 'mrs\.', 'lect\.dr\.',
      'dr', 'professor', 'mr', 'mrs', 'ing\.'
    ]

    for t in titles:
      if re.match(t, self.tkn, re.IGNORECASE):
        return 1
    return 0

  def get_parent(self, element):
    if element is None:
      return 'none'
    if not element.parent is None:
      return element.parent.name
    return 'none'

  def get_second_parent(self, element):
    if element is None:
      return 'none'
    if element.parent != None:
      if element.parent.parent != None:
        return element.parent.parent.name
    return 'none'

  def get_class_name(self, element):
    if element is None:
      return 'none'
    while not element is None:
      if element.has_attr("class"):
        try:
          class_name = '.'.join(element.get("class"))
          return class_name
        except:
          break
        break
      element = element.parent
    return 'none'

  def get_text_depth(self, element):
    if element is None:
      return 'none'
    text_depth = 0
    while not element is None:
      element = element.parent
      text_depth += 1
    return text_depth

  def get_element_position(self, element):
    if element is None:
      return 'none'
    element_position = 0
    if not element.parent is None:
      for child in element.parent.findChildren():
        if child == element:
          break
        element_position += 1
    return element_position

  def set_features(self):
    html_context = self.get_parent(self.element) + '.' + self.get_second_parent(self.element)

    self.features = [
      str(remove_accents(self.tkn)),
      0, # Exact match.
      0, # Partial match.
      0, # Name gazetteer count.
      0, # Word gazetteer count.
      str(self.is_email()),
      str(self.is_number()),
      str(self.is_title()),
      str(self.is_url()),
      str(self.is_capitalized()),
      str(self.is_punctuation()),
      str(html_context), 
      str(self.get_class_name(self.element))
    ]

class Tokenizer():
  def __init__(self):
    self.exact_gazetteer = {}
    self.partial_gazetteer = {}

  def create_token(self, tkn_value, element):
    return HtmlToken(tkn_value, element)

  def assign_correct_labels(self, tokens, correct_names):
    correct_names = [n.split(' ') for n in correct_names]

    i = 0
    while i < len(tokens):
      size = 0
      for name in correct_names: 
        match = True
        for j in range(0, len(name)):
          if i + j >= len(tokens) or tokens[i + j].tkn != name[j]:
            match = False
            break
        if match:
          size = len(name)  
          break

      if size == 0:
        i += 1
      else:
        # prev_label = 'O'
        # if i > 0:
        #   prev_label = tokens[i-1].bio_tag
        # #   if i > 1 and prev_label == 'PUNCT':
        # #     prev_label = tokens[i-2].bio_tag

        # if prev_label == 'I-PER':
        #   tokens[i].bio_tag = 'B-PER'
        # else:
        #   tokens[i].bio_tag = 'I-PER'

        # # tokens[i].bio_tag = 'B-PER'
        # for j in range(i+1, i + size):
        #   tokens[j].bio_tag = 'I-PER'
        #   # if not is_punctuation(tokens[j].tkn):
        #   #  tokens[j].bio_tag = 'I-PER'

        prev_label = 'O'
        if i > 0:
          prev_label = tokens[i-1].bio_tag

        if prev_label == 'I-PER':
          tokens[i].bio_tag = 'B-PER'
        else:
          tokens[i].bio_tag = 'I-PER'

        for j in range(i+1, i + size):
          tokens[j].bio_tag = 'I-PER'

        i += size

    for i in range(len(tokens)):
      # if is_punctuation(tokens[i].tkn):
      #   tokens[i].bio_tag = 'PUNCT'

      # Partial match.
      if tokens[i].tkn in self.partial_gazetteer:
        tokens[i].features[2] = 1
        tokens[i].features[3] = round(log(self.partial_gazetteer[tokens[i].tkn]))
      elif is_punctuation(tokens[i].tkn):
        tokens[i].features[2] = 1

      tkn = tokens[i].features[0]

      # Exact match.
      for j in reversed(range(6)): 
        if i + j >= len(tokens) or j == 0:
          continue

        name = [t.tkn.lower() for t in tokens[i:i+j+1] if not is_punctuation(t.tkn)] 
        if len(name) <= 1:
          continue

        # match = False
        # for rotation in range(j):
        n = ' '.join(name)
        if n in self.exact_gazetteer:
          # match = True
          for k in range(i, i+j+1):
            tokens[k].features[1] = 1
          break

        # if match:
        #   break

  def get_block_element(self, tkn):
    element = tkn.element

    tags = ['span', 'em', 'td', 'a', 'strong', 'b', 'font', 'sup']
    while element != None:
      if not element.name in tags: 
        return element
      element = element.parent
    return None

  def split_sentence(self, sentence):
    if len(sentence) < 50:
      return [sentence]

    s = []
    sentences = []
    for i in range(len(sentence) - 1):
      s.append(sentence[i])
      if sentence[i].tkn.endswith('.'):
        if sentence[i+1].tkn[0].isupper():
          sentences.append(s)
          s = []
    s.append(sentence[-1])
    sentences.append(s)
    return sentences

  def tokenize(self, html, correct_names=[]):
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script and style tags.
    [s.extract() for s in soup('script')]
    [s.extract() for s in soup('style')]
    for br in soup.find_all("br"):
      if len(br.findChildren()) == 0:
        br.string = 'BR'

    # Iterate through all text elements.
    tkns = []
    n_strs = [i for i in soup.recursiveChildGenerator() if type(i) == NavigableString]
    for n_str in n_strs:
      content = n_str.strip()
      if len(content) == 0:
        continue

      for t in tokenize_text(content):
        t = self.create_token(t, n_str.parent)
        tkns.append(t)

    if len(correct_names) > 0:
      self.assign_correct_labels(tkns, correct_names)

    el = None 
    sentences = []
    sentence = []
    for t in tkns:
      next_el = self.get_block_element(t)
      if el is None:
        el = next_el
      elif el == next_el:
        pass
      else:
        el = next_el
        if len(sentence) > 0:
          sentences += self.split_sentence(sentence)
        sentence = []

      if el.name != 'br':
        sentence.append(t)

    if len(sentence) > 0:
      sentences += self.split_sentence(sentence)

    for s in sentences:
      if len(s) > 0:
        if s[0].bio_tag == 1:
          s[0].bio_tag = 2

    return sentences

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
      name = " ".join(tokenize_text(name.strip()))
      target_names.append(name.strip())

  with open('data/gazetteer.txt') as f:
    for name in f:
      if len(name) == 0: 
        continue

      name = name.strip()
      exact_name = []
      for t in tokenize_text(name):
        exact_name.append(t.lower())
        if not t in tokenizer.partial_gazetteer:
          tokenizer.partial_gazetteer[t] = 0
        tokenizer.partial_gazetteer[t] += 1
        
      exact_name = ' '.join(exact_name)
      if not exact_name in tokenizer.exact_gazetteer:
        tokenizer.exact_gazetteer[exact_name] = 0
      tokenizer.exact_gazetteer[exact_name] += 1
  
  with open(html_file) as f:
    sentences = tokenizer.tokenize(f.read(), target_names)
  
    for i, s in enumerate(sentences):
      for t in s:
        print(t.tkn, t.bio_tag, ' '.join(str(t) for t in t.features))
      if i < len(sentences) - 1:
        print('')
