
# coding: utf-8

# In[194]:


'''overall plan of action 
# 1. take a small subset of the train dataset
# 2. clean the data - remove all stop-words -- done
# 3. convert the text to a dictionary (features)
# 4. from the created dictionary, identify the keywords (if the keyword is found and the train data say 1 then identify it)
# 5. for all the tweets, where any of the identified keywords are found, mark them
# 6. calculate the precision and accuracy
'''


# In[195]:


# import the modules needed
import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[196]:


dframe = pd.read_csv('train_data.csv')


# In[197]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))


# In[198]:


final_text = []
for row in dframe.tweet:
    tweet_text = row
    tweet_text = ''.join(i for i in tweet_text if ord(i)<128)
    word_tokens = word_tokenize(tweet_text)
    new_text = [w for w in word_tokens if not w in stop_words]
    new_text = []
    for w in word_tokens:
        if w not in stop_words:
            new_text.append(w)
            
    str1 = ' '.join(new_text)
    final_text.append(str1)


# In[199]:


dframe['filtered_tweet'] = pd.DataFrame({'filter':final_text})


# In[200]:


dframe.head()


# In[186]:


# for all cases where the label = 1, get the text of 'new' and convert to a dictoionary

