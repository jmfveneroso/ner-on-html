#!/usr/bin/python
# coding=UTF-8

import np
import re

def is_punctuation(text):
  return re.match("^[\,\;\:\-\"\(\)“”]$", text)

def gazetteer_predict(X, T, mode='partial'):
  Y = np.zeros((X.shape[0], X.shape[1], 3))

  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      if T[i,j] is None:
        continue

      Y[i,j] = np.array([1, 0, 0])
      # if is_punctuation(T[i,j]):
      #   Y[i,j] = np.array([0, 0, 1])

      # Exact.
      if mode == 'exact':
        if X[i,j,0] == 1:
          Y[i,j] = np.array([0, 0, 1])

      # Partial .
      else:
        if X[i,j,1] == 1:
          Y[i,j] = np.array([0, 1, 0])
  return Y      
