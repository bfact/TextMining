import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))[:5000]
print "done"

## 1
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  unigrams = r.split()
  bigrams = zip(unigrams[:-1],unigrams[1:])
  for w in bigrams:
    wordCount[w] += 1

print len(wordCount)
## 182246

# top five
sorted_bigrams = [(wordCount[w], w) for w in wordCount]
sorted_bigrams.sort()
sorted_bigrams.reverse()
print sorted_bigrams[:5]

# [(4587, ('with', 'a')), (2595, ('in', 'the')), 
#  (2245, ('of', 'the')), (2056, ('is', 'a')), 
#  (2033, ('on', 'the'))]

## 2 

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:1000]]

### Sentiment analysis

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  unigrams = r.split()
  bigrams = zip(unigrams[:-1],unigrams[1:])
  for w in bigrams:
    if w in words:
      feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]


theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
print residuals

# array([ 1715.53738858])


## 3

# get counts for unigrams
u_wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  unigrams = r.split()
  for w in unigrams:
    u_wordCount[w] += 1

    
u_counts = [(u_wordCount[w], w) for w in u_wordCount]
u_counts.sort()
u_counts.reverse()

u_words = [x[1] for x in u_counts[:1000]]

# bigram + unigram
ub_counts = counts + u_counts
ub_counts.sort()
ub_counts.reverse()

# 1000 features as combination of unigrams and bigrams
ub_words = [x[1] for x in ub_counts[:1000]]

print u_words[:5]
print words[:20]
print ub_words[:20]

### Combined Sentiment analysis

u_wordId = dict(zip(ub_words, range(len(ub_words))))
u_wordSet = set(ub_words)

def feature2(datum):
  feat = [0]*len(ub_words)
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  unigrams = r.split()
  bigrams = zip(unigrams[:-1],unigrams[1:])
  for w in bigrams:
    if w in ub_words:
      feat[u_wordId[w]] += 1
  for w in unigrams:
    if w in ub_words:
      feat[u_wordId[w]] += 1
  feat.append(1) #offset
  return feat

X = [feature2(d) for d in data]
y = [d['review/overall'] for d in data]

theta2,residuals,rank,s = numpy.linalg.lstsq(X, y)
print residuals

# array([ 1447.41983182])



## 4
zipped = zip(theta2, ub_words)
zipped.sort()
negWeights = zipped[:5]
posWeights = zipped[len(zipped)-5:]

print posWeights
print negWeights

# [(0.21414947317083155, ('the', 'best')), (0.21966121359450433, ('not', 'bad')), 
#  (0.22589518336319833, ('of', 'these')), (0.23007899372560339, ('a', 'bad')), 
#  (0.70239362980437137, 'sort')]

# [(-0.83079795326631456, ('sort', 'of')), (-0.27419146505544234, 'water'), 
#  (-0.23888121677224239, 'corn'), (-0.23013855443374615, ('the', 'background')), 
#  (-0.20651255101042032, 'kind')]

## 5

# unigram predictor values
zipped1 = zip(theta3, unigram_words)
zipped1.sort()
negWeights1 = zipped1[:5]
posWeights1 = zipped1[len(zipped1)-5:]

print negWeights1
print posWeights1

# [(-0.38419429248466058, 'skunk'), (-0.3280567765234913, 'skunky'), 
#  (-0.32646097412740971, 'bland'), (-0.29137707362423787, 'oh'), 
#  (-0.25104187735801958, 'water')]

# [(0.17953565223020301, 'summer'), (0.17968841874159203, 'impressed'), 
#  (0.18160569739312082, 'wonderful'), (0.25065984633199245, 'always'), 
#  (0.25226326129579446, 'exceptional')]

# bigram predictor values
zipped2 = zip(theta, words)
zipped2.sort()
negWeights2 = zipped2[:5]
posWeights2 = zipped2[len(zipped2)-5:]

print negWeights2
print posWeights2

# [(-0.46338159790801697, ('ton', 'of')), (-0.33959601846800014, ('just', 'not')), 
#  (-0.33687338516838161, ('not', 'very')), (-0.30659892462695088, ('pale', 'yellow')),
#  (-0.28483311948206747, ('it', 'goes'))]

# [(0.24393136874196836, ('will', 'be')), (0.24674168037461863, ('is', 'smooth')), 
#  (0.29339620528772048, ('very', 'drinkable')), (0.3025614994475056, ('it', 'pours')), 
#  (0.43091425667858801, ('a', 'ton'))]

## 6
import operator
import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

bloblist = []
for d in data:
  review = d['review/text']
  bloblist.append(tb(review))

# foam
foam_idf = idf('foam', bloblist)
foam_tfidf = tfidf('foam', bloblist[0], bloblist)

print foam_idf
print foam_tfidf

# 0.955460239608
# 0.0389983771268

# smell
smell_idf = idf('smell', bloblist)
smell_tfidf = tfidf('smell', bloblist[0], bloblist)

print smell_idf
print smell_tfidf

# 0.571217488503
# 0.0116574997654

# lactic
lactic_idf = idf('lactic', bloblist)
lactic_tfidf = tfidf('lactic', bloblist[0], bloblist)

print lactic_idf
print lactic_tfidf

# 2.85387196432
# 0.116484569972

# tart
tart_idf = idf('tart', bloblist)
tart_tfidf = tfidf('tart', bloblist[0], bloblist)

print tart_idf
print tart_tfidf

# 1.00966114521
# 0.0206053294941


## 7

review1 = ''.join([c for c in bloblist[0].lower() if not c in punctuation])
review2 = ''.join([c for c in bloblist[1].lower() if not c in punctuation])

unigrams1 = set(review1.split())
unigrams2 = set(review2.split())

newset = set(unigrams1).intersection(unigrams2)
    
dataSet1 = defaultdict(int)
dataSet2 = defaultdict(int)
for w in review1.split():
  if w in newset:
    dataSet1[w] += 1

for w in review2.split():
  if w in newset:
    dataSet2[w]+=1

d1_tfidf = []
d2_tfidf = []
for key in newset:
  d1_tfidf.append(tfidf(key,bloblist[0],bloblist))
  d2_tfidf.append(tfidf(key,bloblist[1],bloblist))

cosineSimilarity = 1 - scipy.spatial.distance.cosine(d1_tfidf, d2_tfidf)

print cosineSimilarity

# 0.911677823526



