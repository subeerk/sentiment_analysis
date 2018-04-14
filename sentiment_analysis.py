
# coding: utf-8

# overall plan of action 
# 1. take a small subset of the train dataset
# 2. clean the data - remove all stop-words
# 3. convert the text to a dictionary (features)
# 4. from the created dictionary, identify the keywords (if the keyword is found and the train data say 1 then identify it) .. done
# 5. for all the tweets, where any of the identified keywords are found, mark them
# 6. calculate the precision and accuracy

# import the modules needed
import numpy as np
import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

# reading the input files
dframe = pd.read_csv('train_data.csv')

# removing other frequently used words in Tweets
# source of these words picked from http://techland.time.com/2009/06/08/the-500-most-frequently-used-words-on-twitter/
# read the custom stop-words from a txt file and keep that in memory
file_handle = open("custom_stop_words.txt","r")
custom_stop_words = file_handle.read()

final_text = []
for row in dframe.tweet:
    tweet_text = row
    tweet_text = ''.join(i for i in tweet_text if ord(i)<128)
    word_tokens = word_tokenize(tweet_text)
    new_text = [w for w in word_tokens if not w in stop_words]
    new_text = []
    for w in word_tokens:
        if w not in stop_words:
            if w not in custom_stop_words:
                new_text.append(w)
            
    str1 = ' '.join(new_text)
    final_text.append(str1)

dframe['filtered_tweet'] = pd.DataFrame({'filter':final_text})
# get the actual list of tweets for which the label = 1
df_label1 = dframe[(dframe.label == 1)]

# get the list of the rows with label = 1 i.e. where the words are taken to be analyzed
# CAVEAT: the created dataframe has a lot of blank rows.
# dframe_label_1 = dframe.mask(lambda x: x['label'] == 0)

# merge all filtered tweets together to create a master list
merged_tweet = []
for row in dframe.filtered_tweet:
    str1 = ''.join(row)
    merged_tweet.append(str1)
    merged_tweet.append(" ")
    
merged_tweet = ' '.join(merged_tweet)

# remove all punctuation marks and single characters added.
clean_tweet = merged_tweet.translate(None, ",.;:@#(?!&$^-_)")

# get the unique word list from all the filtered tweets that are marke
from string import punctuation
unique_keywords = [w for w in set(clean_tweet.translate(None, punctuation).lower().split()) if len(w) >= 1]
text_file = open("unique_words.txt", "w")
text_file.write("%s" %unique_keywords)
text_file.close()

## Model training complete. 
## The output of this is a file named "unique_words.txt" which has
## the list of the unique words


dframe_test = pd.read_csv('test_data.csv')
# read in the overall unique words
file_handle_uq = open("unique_words.txt","r")
unique_words = file_handle_uq.read()

#result = ""
#for row in dframe.tweet:
#    tweet_text = row
#    tweet_text = ''.join(i for i in tweet_text if ord(i)<128)
    # match each word of the tweet_text with the unique_words
    # if any of the word matches, then mark that tweet value 
    # as 1, else it would be 0
#    for w in tweet_text.split:
#        if w in unique_words:
#           result  
