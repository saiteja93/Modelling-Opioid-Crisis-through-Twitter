#!/usr/bin/python 2.7
# -*- coding: utf8 -*-

#Author - Saiteja Sirikonda
#Course : Natural Language Processing, Spring 2018 taught by Prof. Chitta Baral


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
import nltk.classify
import nltk
import json, string

#Drug keywords that indicate Opioid use
array = ["morphine", "methadone", "buprenorphine", "hydrocodone", "oxycodone","heroin", "oxycontin", "perc", "percocet","palladone" , "vicodin", "percodan", "tylox" ,"demerol", "oxy", "roxies","opiates", "oxy", "percocet", "percocets", "hydrocodone", "norco",
	    	"norcos", "roxy", "roxies", "roxycodone", "roxicodone", "opana", "opanas", "prozac", "painrelief", "painreliever", "painkillers", "addiction", "opium"]
	    	

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


#open the JSON to a list of dictionaries
with open("Merged_Labelled.json", "r") as f:
	original_data = json.load(f)
#print type(data)

# print data[:5]
# print ("\n")
# print ("\n")

# i = data[0]
# print data[0]
# print ("\n")
# print i["tweet"], i["label"]
# print len(data)
#print data[0]
# print "the actual JSON", data
# print "\n"
data = original_data[:3]
#print data["tweet"]
punctuation = "!()-[]{};@#$%?~^&*"
words = {}
for i in stopwords.words("english"):
	k = i.encode("utf8")
	words[k] = 0

for i in xrange(len(data)):
	#splitting out each label and tweet
	temp_tweet, temp_label = "", ""
	temp_tweet = data[i]["tweet"].lower().encode("utf8").replace(".","").replace("!","").replace(",","")
	temp_label = data[i]["label"].encode("utf8")
	# print "extract tweet", temp_tweet
	# print "extract label", temp_label
	# print "\n"
	#print "look for capitals", temp
	#removing the punctuation marks
	#temp = t.translate(string.maketrans("",""),string.punctuation)
	data.pop(i)
	temp1 = []
	temp1 = temp_tweet.split(" ")
	#print "list form", temp1
	#Removing the stop words
	temp1 = [x for x in temp1 if x not in words]
	print temp1
	#print temp1
	# print "modified", temp1
	# print "\n"
	# #Replacing drug instances with "Druginstance" keyword
	for i,j in enumerate(temp1):
		# if "\n" in j:
		# 	new_j = ""
		# 	for k in j:
		# 		if k not in punctuation:
		# 			new_j +=k
		# 		else: break
		# 	if new_j in array:
		# 		temp1[i] = "druginstance" + "\n"
		if j in array:
			temp1[i] = "druginstance"
	print "after drug instance", temp1
	# print "\n"
	temp_tweet = " ".join(temp1)
	temp = {"tweet".decode("utf8"): temp_tweet.decode("utf8"), "label".decode("utf8"):temp_label.decode("utf8")}
	#print "string form again", temp
	print "final", temp
	print "\n"
	data.insert(i,temp)
	raw_input("please press enter ..")

print data

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
