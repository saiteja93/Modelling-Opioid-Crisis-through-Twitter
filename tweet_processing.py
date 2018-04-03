#!/usr/bin/python 2.7
# -*- coding: utf8 -*-


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
import nltk.classify
import nltk
import json


# # list of text documents

# text = ["does anybody delivers oxycodone ???plssss reply oxycodone",
# 		"we'll get her straight on the Prozac.",
# 		"Oxycodone is the shit",
# 		"The last time I missed my Prozac my OCD shot through the roof and I was on 20mg and now I'm on 40mg oh god today isn't good",
# 		"Oxycodone doing some great things to my wounds right now"]

#text = ['This is the first document.','This is the second second document.','And the third one.','Is this the first document?']
#text = ["The quick brown fox jumped fox over the lazy dog."]

#text = ["we'll get her straight on the Prozac."]
# text = []
# with open("Merged_Labelled.json", 'r') as f:
# 	dictionary = json.load(f)

# for dic in dictionary:
# 	text.append(dic['tweet'])


# print len(text)

# count_vect = CountVectorizer(ngram_range = (1,2), min_df = 1)

# X_train = count_vect.fit_transform(text).toarray()
# #print X_train

# index = count_vect.vocabulary_.get("reply oxycodone")
# print X_train[:,index]
# #print "saiteja"





# print X_train.toarray()
# print X_train.shape

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



with open("Merged_Labelled.json", "r") as f:
	data = json.load(f)
#print type(data)

# print data[:5]
# print ("\n")
# print ("\n")

# i = data[0]
# print data[0]
# print ("\n")
# print i["tweet"], i["label"]
# print len(data)


words = {}
for i in stopwords.words("english"):
	k = i.encode("utf8")
	words[k] = 0

for i in xrange(len(data)):
	temp_tweet = data[i]["tweet"].lower().encode("utf8")
	temp_label = data[i]["label"].encode("utf8")
	#print "look for capitals", temp
	#removing the punctuation marks
	#temp = t.translate(string.maketrans("",""),string.punctuation)
	data.pop(i)
	temp1 = temp_tweet.split(" ")
	#print "list form", temp1
	temp1 = [x for x in temp1 if x not in words]
	#print temp1
	#print "modified array element", temp1
	temp_tweet = " ".join(temp1)
	temp = {"tweet": temp_tweet.decode("utf8"), "label":temp_label.decode("utf8")}
	#print "string form again", temp
	data.insert(i,temp)


#the first 90% data is for training and the rest for test
cut_off = int (len(data) * 0.90)

train_data = data[:cut_off]
test_data = data[cut_off:]

#print train_data[0]
#print len(train_data), len(test_data)

def feature_extraction(tweet_row):
	tweet_features = {}

	tweet = tweet_row["tweet"].lower().replace("\n","")
	for trigrams in nltk.trigrams(tweet.split(" ")):
		tweet_features["presence(%s,%s,%s)" % (trigrams[0],trigrams[1],trigrams[2])] = True

	return tweet_features


train_set = [(feature_extraction(d),d["label"]) for d in train_data]
test_set = [(feature_extraction(d),d["label"]) for d in test_data]


# classifier = nltk.NaiveBayesClassifier

# classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier = nltk.classify.SklearnClassifier(LinearSVC())
classifier.train(train_set)

#classifier.show_most_informative_features(20)

# collect tweets that were wrongly classified
errors = []
for d in test_set:
    label = d[1]
    guess = classifier.classify(d[0])
    print guess
    #print "guess", guess, "label", label
    if guess != label:
        errors.append( (label, guess, d) )

# for (label, guess, d) in sorted(errors):
#     print 'correct label: %s\nguessed label: %s\ntweet=%s\n' % (label, guess, d['tweet'])

print 'Total errors: %d' % len(errors)

print 'Accuracy: ', nltk.classify.accuracy(classifier, test_set)













	