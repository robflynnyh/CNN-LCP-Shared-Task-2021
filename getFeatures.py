import math
import numpy as np 
import pandas as pd 

def getFreq(_string, freqDict):
  length = len(_string)
  totals = []
  for i in range(length):
    if(_string[i] in freqDict):
      currVal = freqDict[_string[i]][0]
      if(math.isnan(currVal)==False):
        totals.append(currVal) ##logFreq
      else:
          totals.append(0.33) ##~For missing values 
    else:
      totals.append(0.33) 
  avg = sum(totals)/len(totals)
  return float(avg)

def getAoA(_string, AoADict):
  _string = _string.split()
  length = len(_string)
  totals = []
  for i in range(length):
    if(_string[i] in AoADict):
      currVal = AoADict[_string[i]][0]
      if(math.isnan(currVal)==False):
        totals.append(currVal)
      else:
          totals.append(1) ##~For missing values 
    else:
      totals.append(1)
  avg = sum(totals)/len(totals)
  return float(avg)

def getsyll(_string, syllDict):
  _string = _string.split()
  length = len(_string)
  totals = []
  for i in range(length):
    if(_string[i] in syllDict):
      currVal = syllDict[_string[i]][0]
      if(math.isnan(currVal)==False):
        totals.append(currVal) 
      else:
          totals.append(0.44) ##~For missing values 
    else:
      totals.append(0.44)
  avg = sum(totals)/len(totals)
  return float(avg)

def getletters(_string):
    _string = _string.split()
    length = len(_string)
    lettotal = 0
    for i in _string:
        lettotal += (len(i)/10) #divide length by 10
    lettotal = lettotal/length
    return lettotal

def getCharEmb(_string, c2): ##c2 = char2vec model
    _string = _string.split()
    embs = c2.vectorize_words(_string).tolist()
    embs = [float(sum(v))/len(v) for v in zip(*embs)] ##average embedding across the provided string
    return embs   

def getEmb(_string, embDict): #embdict = Dictionary containing 50d GloVe embeddings
  _string = _string.split()
  embs = []
  for word in _string:
    if word in embDict:
      current = embDict[word]
    else:
      current = [0]*50 ##Zero vectoe
    embs.append(current) 
  embs = [float(sum(v))/len(v) for v in zip(*embs)]
  return embs

