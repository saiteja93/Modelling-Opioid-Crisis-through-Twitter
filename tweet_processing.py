#!/usr/bin/python 2.7
# -*- coding: utf8 -*-


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import string
import nltk


# # list of text documents
# text = ["does anybody delivers oxycodone ???plssss reply oxycodone",
# 		"we'll get her straight on the Prozac.",
# 		"Oxycodone is the shit",
# 		"The last time I missed my Prozac my OCD shot through the roof and I was on 20mg and now I'm on 40mg oh god today isn't good",
# 		"Oxycodone doing some great things to my wounds right now"]
#text = ["The quick brown fox jumped fox over the lazy dog."]

#text = ["we'll get her straight on the Prozac."]
# ...
#Creating the Hashmap of stopwords taken from the nltk library
words = {}
for i in stopwords.words("english"):
	k = i.encode("utf8")
	words[k] = 0
#print words


#removing stopwords from each tweet
for i in xrange(len(text)):
	temp = text[i].lower()
	print "look for capitals", temp
	#removing the punctuation marks
	#temp = t.translate(string.maketrans("",""),string.punctuation)
	text.pop(i)
	temp1 = temp.split(" ")
	print "list form", temp1
	temp1 = [x for x in temp1 if x not in words]
	print temp1
	#print "modified array element", temp1
	temp = " ".join(temp1)
	#print "string form again", temp
	text.insert(i,temp)
print text







count_vect = CountVectorizer(analyzer = nltk.trigrams, ngram_range = (3,3))
count_vect.fit(text)
X_train = count_vect.transform(text)


print X_train.toarray()
print X_train.shape

# tfidf_transformer = TfidfTransformer()
# X_train_tidf = tfidf_transformer.fit_transform(X_train)
# print X_train_tidf
# print X_train_tidf.shape
# print X_train_tidf.toarray()
# #create the transform
# vectorizer = TfidfVectorizer()

# # tokenize and build vocab
# vectorizer.fit(filtered_words)
# # summarize
# print(vectorizer.vocabulary_)
# print(vectorizer.idf_)
# # encode document
# vector = vectorizer.transform([text[0]])
# # summarize encoded vector
# print(vector.shape)
# print(vector.toarray())

	