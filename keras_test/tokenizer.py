#!/usr/bin/python
# coding=UTF-8

import sys
import re
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

class HtmlToken():
  """ 
  An HTML token is a composite object containing a value, and the corresponding HTML element from where that value was extracted.
  It also holds a list of textual and structural features.
  """

  def __init__(self, tkn, element):
    self.tkn = tkn
    self.element = element
    self.features = []
    self.bio_tag = 'O'
    self.exact_match = False
    self.partial_match = False

  def set_features(self, features):
    self.features = features

class Tokenizer():
  def __init__(self):
    self.exact_gazetteer = {}
    self.partial_gazetteer = {}

  def tokenize_text(self, text):
    return re.compile("\s+").split(text)

  def create_token(self, tkn_value, element):
    tkn = HtmlToken(tkn_value, element)
    tkn.set_features([
      str(self.get_parent(element)), 
      str(self.get_second_parent(element)), 
      str(self.get_class_name(element)), 
      str(self.get_text_depth(element)), 
      str(self.get_element_position(element))
    ])

    return tkn

  def get_parent(self, element):
    """ 
    Returns the parent tag of an HTML element.
    """
    if element.parent != None:
      return element.parent.name
    return ''

  def get_second_parent(self, element):
    """ 
    Returns the second parent tag of an HTML element.
    """
    if element.parent != None:
      if element.parent.parent != None:
        return element.parent.parent.name
    return ''

  def get_class_name(self, element):
    while element != None:
      if element.has_attr("class"):
        try:
          class_name = ".".join(element.get("class"))
          return class_name
        except:
          break
        break
      element = element.parent
    return 'none'

  def get_text_depth(self, element):
    """ 
    Returns the number of "indents" in the DOM Tree until the HTML
    element is reached.
    """
    text_depth = 0
    while element != None:
      element = element.parent
      text_depth += 1
    return text_depth

  def get_element_position(self, element):
    """ 
    Returns the position of an HTML element in comparison with its siblings 
    relative to the HTML parent element.
    """
    element_position = 0
    if element.parent != None:
      for child in element.parent.findChildren():
        if child == element:
          break
        element_position += 1
    return element_position

  def assign_correct_labels(self, tokens, correct_names):
    correct_names = [n.split(' ') for n in correct_names]
    # print(correct_names)

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
        tokens[i].bio_tag = 'B-NAME'
        for j in range(i + 1, i + size):
          tokens[j].bio_tag = 'I-NAME'
        i += size

    for i in range(len(tokens)):
      # Partial match.
      if tokens[i].tkn in self.partial_gazetteer:
        tokens[i].partial_match = True

      # Exact match.
      for j in reversed(range(6)): 
        if i + j >= len(tokens):
          continue

        name = ' '.join([t.tkn for t in tokens[i:i+j+1]])
        if name in self.exact_gazetteer:
          for k in range(i, i+j+1):
            tokens[k].exact_match = True
          break

  def get_block_element(self, tkn):
    element = tkn.element

    tags = ['span', 'em', 'td', 'a']
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

      for t in self.tokenize_text(content):
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
 
    return sentences

if __name__ == '__main__':
  if len(sys.argv) != 4:
    print('Wrong arguments')
    quit()

  tokenizer = Tokenizer()
  
  target_names = []
  with open(sys.argv[2]) as f:
    for name in f:
      if len(name) == 0: 
        continue
      name = " ".join(tokenizer.tokenize_text(name.strip()))
      target_names.append(name.strip())

  with open(sys.argv[3]) as f:
    for name in f:
      if len(name) == 0: 
        continue

      name = name.strip()
      tokenizer.exact_gazetteer[name] = True
      for t in name.split(' '):
        tokenizer.partial_gazetteer[t] = True
  
  with open(sys.argv[1]) as f:
    sentences = tokenizer.tokenize(f.read(), target_names)
  
    for i, s in enumerate(sentences):
      for t in s:
        print(t.tkn, t.bio_tag, 1 if t.exact_match else 0, 1 if t.partial_match else 0, ' '.join(t.features))
      if i < len(sentences) - 1:
        print('')
