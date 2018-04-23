# -*- coding: utf8 -*-

#Author - Saiteja Sirikonda
#Course : Natural Language Processing, Spring 2018 taught by Prof. Chitta Baral

import numpy as np
import pandas as pd
import re, json, nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.deprecated.doc2vec import LabeledSentence
LabeledSentence = gensim.models.deprecated.doc2vec.LabeledSentence

array = ["morphine", "methadone", "buprenorphine", "hydrocodone", "oxycodone","heroin", "oxycontin", "perc", "percocet","palladone" , "vicodin", "percodan", "tylox" ,"demerol", "oxy", "roxies","opiates", "oxy", "percocet", "percocets", "hydrocodone", "norco",
	    	"norcos", "roxy", "roxies", "roxycodone", "roxicodone", "opana", "opanas", "prozac", "painrelief", "painreliever", "painkillers", "addiction", "opium"]

words = {}
for i in stopwords.words("english"):
	k = i.encode("utf8")
	words[k] = 0

# regular expressions used to clean up the tweet data
drug = re.compile('|'.join(array).lower())
http_re = re.compile(r'\s+http://[^\s]*')
remove_ellipsis_re = re.compile(r'\.\.\.')
at_sign_re = re.compile(r'\@\S+')
punct_re = re.compile(r"[\"'\[\],.:;()\-&!]")
price_re = re.compile(r"\d+\.\d\d")
number_re = re.compile(r"\d+")

def normalize_tweet(tweet):
    t = tweet.lower()
    t = re.sub(price_re, 'PRICE', t)
    t = re.sub(remove_ellipsis_re, '', t)
    t = re.sub(drug, 'druginstance', t)
    t = re.sub(http_re, ' LINK', t)
    t = re.sub(punct_re, '', t)
    t = re.sub(at_sign_re, '@', t)
    t = re.sub(number_re, 'NUM', t)
    return t

def feature_extractor(tweet):
	#remove new lines
	features_per_tweet = {}
	new_tweet = tweet.strip().lower().encode('ascii', errors='ignore')
	new_tweet = normalize_tweet(new_tweet)
	#print new_tweet
	#remove new lines (\n)
	#new_tweet = re.sub(r"\n", " ", new_tweet)
	words_in_tweet = new_tweet.split(" ")
	words_in_tweet = [x for x in words_in_tweet if x not in words]
	# for i,j in enumerate(words_in_tweet):
	# 	if j in array:
	# 		words_in_tweet[i] = "druginstance"
	new_tweet = " ".join(words_in_tweet).decode("ascii", errors = "ignore")
	#print new_tweet
	#print ("\n")

	tokens = tokenizer.tokenize(new_tweet)
	
	#raw_input("please press enter ...")

	return tokens



#open the JSON to a list of dictionaries
with open("Merged_Labelled.json", "r") as f:
	original_data = json.load(f)
data = original_data

#print data[0]
#Turning it into a DataFrame
# cut_off = int (len(data) * 0.90)

# train_data = data[:cut_off]
# test_data = data[cut_off:]

formatted_data = [(d["label"],feature_extractor(d["tweet"])) for d in data]
#test_set = [(d["label"],feature_extractor(d["tweet"])) for d in test_data]

df = pd.DataFrame(formatted_data, columns = ["sentiment", "tweet"])

#print df.head(10)
#removing the itemid
#df.drop("index", inplace = True, axis = 1)

x_train, x_test, y_train, y_test = train_test_split(np.array(df.tweet),
                                                    np.array(df.sentiment), test_size=0.2)


def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

#print x_train[3][1]

tweet_w2v = Word2Vec(size = 200, window = 10, min_count=5, workers = 11, alpha = 0.025, iter = 20)
tweet_w2v.build_vocab([x[0] for x in x_train])
m = tweet_w2v.corpus_count
#print m
tweet_w2v.train([x[0] for x in x_train], epochs = tweet_w2v.iter, total_examples = 2294)


#print tweet_w2v["druginstance"]

final_embedding = tweet_w2v._nemb_final.eval()

