Team #5: Saiteja Sirikonda, Sidharth Panicker, Sujitha Metla, Chandana Manjunath Ningappa and Namitha Sudheendra
COurse: Natural Language Processing

This aim of the project is to Model the Dynamics of the Opioid Crisis through Twitter. 

Dealing with the Data: 
When we found corpus for the drug-related tweets, they had tweets, with mentions about all sorts of drugs to it. But, for the sake of this project we are only interested in tweets which are related to Opioids. We looked for terms that are representative of Opioid usage and came up with the following array and formulated that any tweet which contains these words is Opioid related.

array = ["morphine", "methadone", "buprenorphine", "hydrocodone", "oxycodone" "heroin", "oxycontin", "perc", "percocet","palladone" , "vicodin", "percodan", "tylox" ,"demerol", "oxy", "roxies","opiates", "oxy", "percocet", "percocets", "hydrocodone", "norco", "norcos", "roxy", "roxies", "roxycodone", "roxicodone", "opana", "opanas", "prozac", "painrelief", "painreliever", "painkillers", "addiction", "opium"]

We then labelled them as:
“None”: If the Tweet is not at all Drug related.
“0”: If the tweet is drug related but not, abuse.
“1”: If the tweet is drug related and also abuse.

Extracting Features:
We decided to use the bigram representatiion of the words and for this we decided to use the CountVectorizer of Python's sklearn library. The output of this feature extraction would be fed to the Classifier.

