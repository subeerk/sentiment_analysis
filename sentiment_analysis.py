
# coding: utf-8

'''
The script takes the train_data.csv for training the model. The training
dataset has to be at the same folder location where the script is placed.
Additionally, the script/model would need a file named custom_stop_words.txt
that holds the stop words which would be ignored from the tweets. These stop
words are in addition to the english words as provided by the Python nltk.

Once the model train itself, it generates a file named  unique_words.txt
the file contains the list of unique words against which the model was trained

Once trained, the model reads the test data from test_data.csv to predict
the "is_provocative" values. This would be appended in the Model_results.csv
which holds the complete information.
Header of this result file is as under:
	id	label	tweet	filtered_tweet_dbg	is_provacative

NOTE: The file custom_stop_words is not added to the GitHub, but is kept on the
Gmail servers. 
'''

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
# source of these words picked from http://techland.time.com/2009/06/08/the-

500-most-frequently-used-words-on-twitter/
# https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-

201107/blob/master/data/opinion-lexicon-English/positive-words.txt
# read the custom stop-words from a txt file and keep that in memory
file_handle = open("custom_stop_words.txt","r")
custom_stop_words = file_handle.read()

# cleaning the text using the english stop_words
final_text = []
for row in dframe.tweet:
    tweet_text = row
    tweet_text = ''.join(i for i in tweet_text if ord(i)<128)
    word_tokens = word_tokenize(tweet_text)
    new_text = [w for w in word_tokens if not w in stop_words]
    new_text = []
    for w in word_tokens:
        if w not in stop_words:
            # using the custome_stop_words identified from other sources
            if w not in custom_stop_words:
                new_text.append(w)
            
    str1 = ' '.join(new_text)
    final_text.append(str1)
    
# adding the dataframe with the filtered tweets back to the original dataframe
dframe['filtered_tweet'] = pd.DataFrame({'filter':final_text})

# get the actual list of tweets for which the label = 1
df_label1 = dframe[(dframe.label == 1)]

# merge all filtered tweets together to create a master list
merged_tweet = []
for row in df_label1.filtered_tweet:
    str1 = ''.join(row)
    merged_tweet.append(str1)
    merged_tweet.append(" ")
    
merged_tweet = ' '.join(merged_tweet)

# remove all punctuation marks and single characters added.
clean_tweet = merged_tweet.translate(None, ",.;:@#(?!&$^-_)")

# get the unique word list from all the filtered tweets that are marke
from string import punctuation
unique_keywords = [w for w in set(clean_tweet.translate(None, 

punctuation).lower().split()) if len(w) >= 1]
text_file = open("unique_words.txt", "w")
text_file.write("%s" %unique_keywords)
text_file.close()

# Model training complete. The output of this is a file named
# "unique_words.txt" which has the list of the unique words
file_handle_uq = open("unique_words.txt","r")
file_handle_cu = open("custom_stop_words.txt","r")
unique_words = file_handle_uq.read()
custom_words = file_handle_cu.read()
unique_words = unique_words.translate(None, "[',]")
custom_words = custom_words.split(" ")
for w in custom_words:
    if w in unique_words:
        unique_words = unique_words.replace(w,'')

text_file = open("unique_words.txt", "w")
text_file.write("%s" %unique_keywords)
text_file.close()

dframe_test = pd.read_csv('test_data.csv')
# read in the overall unique words
file_handle_uq = open("unique_words.txt","r")
unique_words = file_handle_uq.read()
unique_words = unique_words.translate(None, "[',]")
unique_words = unique_words.split(" ")

filtered_tweet_dbg = []
model_results = []
model_results_coded = []
for row in dframe_test.tweet:
    tweet_text = ''.join(i for i in row if ord(i)<128)
    tweet_text = tweet_text.translate(None, ",.;:@#([?!&$^-_])")
    tokens = word_tokenize(tweet_text)
    new_text = [w for w in tokens if not w in stop_words]
    new_text = []
    for w in tokens:
        if w not in stop_words:
            # using the custom_stop_words identified from other sources
            if w not in custom_stop_words:
                new_text.append(w)
            
    str1 = ' '.join(new_text)
    # str1 holds the clean tweet - ready to be tested
    # ('father dysfunctional drags dysfunction')
    # ("lyft credit n't n't offer wheelchair vans pdx disapointed getthanked")
    # ('bihday')
    
    # split each word of the cleaned tweet 
    string_directive = ''
    for w in str1.split():
        try:
            index = unique_words.index(w)
            if index > -1:
                string_directive = string_directive + "Y"
        except ValueError:
            string_directive = string_directive + "N"
            
    model_results.append(string_directive)
    if "Y" in string_directive:
        string_directive = "1"
    else:
        string_directive = "0"
    model_results_coded.append(string_directive)
    
    filtered_tweet_dbg.append(str1)

# Uncomment the next line for debugging purpose
# dframe_test['model_results'] = pd.DataFrame({'filter':model_results})
dframe_test['filtered_tweet_dbg'] = pd.DataFrame({'filter':filtered_tweet_dbg})
dframe_test['is_provacative'] = pd.DataFrame({'filter':model_results_coded})
dframe_test.to_csv("Model_results.csv")
