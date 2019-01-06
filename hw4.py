
# coding: utf-8

# # Homework 4: Document Classification
# 
# In this problem, you will implement several text classification systems using the naive Bayes algorithm and semi-supervised Learning. In addition to the algorithmic aspects -- you will implement a  naive Bayes classifier and semi-supervised learning protocols on top of it -- you will also get a taste of data preprocessing and feature extraction needed to do text classification.
# 
# Please note that we are leaving some of the implementation details to you. In particular, you can decide how to representation the data internally, so that your implementation of the algorithm is as efficient as possible. 
# <b>Note, however, that you are required to implement the algorithm yourself, and not use existing implementations, unless we specify it explicitly.</b>
# 
# Also note that you are free to add <b>optinal</b> parameters to the given function headers. However, please do not remove existing parameters or add additional non-optional ones. Doing so will cause Gradescope tests to fail.

# ## 1.1: Dataset
# 
# For the experiments, you will use the 20 newsgroups text dataset. It consists of âˆ¼18000 newsgroups posts on 20 topics. We have provided 2 splits of the data. In the first split, 20news-bydate, we split the training and testing data by time, i.e. you will train a model on the data dated before a specified time and test that model on the data dated after that time. This split is realistic since in a real-world scenario, you will train your model on your current data and classify future incoming posts.

# ## 1.2: Preprocessing
# 
# In the first step, you need to preprocess all the documents and represent it in a form that can be used later by your classifier. We will use three ways to represent the data:
# 
# * Binary Bag of Words (B-BoW)
# * Count Bag of Words (C-BoW)
# * TF-IDF
# 
# We define these representations below. In each case, we define it as a matrix, where rows correspond to documents in the collection and columns correspond to words in the training data (details below). 
# 
# However, since the vocabulary size is too large, it will not be feasible to store the whole matrix in memory. You should use the fact that this matrix is really sparse, and store the document representation as a dictionary mapping from word index to the appropriate value, as we define below. For example, $\texttt{\{'doc1': \{'word1': val1, 'word2': val2,...\},...\}}$
# 
# <b><i>Please name the documents $<folder\_name>/<file\_name>$ in the dictionary.</i></b> i.e. 'talk.politics.misc/178761'
# <b>We would like you to do the preprocessing yourself, following the directions below. Do not use existing tokenization tools.</b>

# ### 1.2.1: Binary Bag of Words (B-BoW) Model
# 
# Extract a case-insensitive (that is, "Extract" will be represented as "extract") vocabulary set, $\mathcal{V}$, from the document collection $\mathcal{D}$ in the training data. Come up with a tokenization scheme - you can use simple space separation or more advanced Regex patterns to do this. You should lemmatize the tokens to extract the root word, and use a list of "stopwords" to ignore words like <i>the</i>, <i>a</i>, <i>an</i>, etc. <b>When reading files, make sure to include the 	exttt{errors='ignore'} option</b>
# 
# The set $\mathcal{V}$ of vocabulary extracted from the training data is now the set of features for your training. You will represent each document $d \in \mathcal{D}$ as a vector of all the tokens in $\mathcal{V}$ that appear in $d$. Specifically, you can think of representing the collection as a matrix $f[d,v]$, defined as follows:
# 
# \begin{equation*}
# \forall ~v \in  \mathcal{V}, ~\forall d \in  \mathcal{D}, ~~~~f[d,v]=
# \begin{cases}
#     1 & \text{if } v \in d \\
#     0 & \text{else}
# \end{cases}
# \end{equation*}
# 
# This should be a general function callable for any training data.

# In[282]:


import os
import math
import heapq
import random
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords
def b_bow(directory_name, use_adv_tokenization=False, useStop = True):    
    '''
    Construct a dictionary mapping document names to dictionaries of words to the 
    value 1 if it appears in the document
    
    :param directory_name: name of the directory where the train dataset is stored
    :param use_adv_tokenization: if False then use simple space separation,
                                 else use a more advanced tokenization scheme
    :return: B-BoW Model
    '''
    #make the dictionaries and define the stopwords
    large_dict = {}
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    #go through each folder and doc
    for folder in os.listdir(directory_name):
        for filename in os.listdir(directory_name + "/" + folder):
            docname = directory_name + "/" + folder + "/" + filename
            file = open(docname, 'r', errors='ignore')
            f_strings = file.read()
            tokenized_words = f_strings.split()
            #filter the bad words out
            if useStop:
                filtered_words = [lemmatizer.lemmatize(word.lower()) 
                                  for word in tokenized_words 
                                  if lemmatizer.lemmatize(word.lower()) not in sw]
            else:
                filtered_words = tokenized_words
            #map word to 1 if it appears in doc
            d_words = {word: 1 for word in filtered_words}
            #add the above dict to the overall dict
            large_dict[folder + "/" + filename] = d_words
            file.close()

        
    return large_dict
#(b_bow("20news-data/20news-bydate-rm-metadata/train"))


# ### 1.2.2: Count Bag of Words (C-BoW) Model
# 
# The first part of vocabulary extraction is the same as above.
# 
# Instead of using just the binary presence, you will represent each document $d \in \mathcal{D}$ as a vector of all the tokens in $\mathcal{V}$ that appear in $d$, along with their counts. Specifically, you can think of representing the collection as a matrix $f[d,v]$, defined as follows:
# 
# \begin{equation*}
# f[d, v] = tf(d, v), ~~~~\forall v \in  \mathcal{V}, ~~~~\forall d \in  \mathcal{D},
# \end{equation*}
# 
# where, $tf(d,i)$ is the <i>Term-Frequency</i>, that is, number of times the word $v \in \mathcal{V}$ occurs in document $d$.

# In[283]:


def c_bow(directory_name, use_adv_tokenization=False, useStop = True):
    '''
    Construct a dictionary mapping document names to dictionaries of words to the 
    number of times it appears in the document
    
    :param directory_name: name of the directory where the train dataset is stored
    :param use_adv_tokenization: if False then use simple space separation,
                                 else use a more advanced tokenization scheme
    :return: C-BoW Model
    '''
    large_dict = {}
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    #go through each folder and doc
    for folder in os.listdir(directory_name):
        for filename in os.listdir(directory_name + "/" + folder):
            docname = directory_name + "/" + folder + "/" + filename
            file = open(docname, 'r', errors='ignore')
            f_strings = file.read()
            tokenized_words = f_strings.split()
            #filter the bad words out
            if useStop:
                filtered_words = [lemmatizer.lemmatize(word.lower()) 
                                  for word in tokenized_words 
                                  if lemmatizer.lemmatize(word.lower()) not in sw]
            else:
                filtered_words = tokenized_words
            
            #map word to its count if it appears in doc            
            d_words = {}
            for word in filtered_words:
                if word in d_words:
                    d_words[word] =  d_words[word]  + 1
                else:
                    d_words[word] = 1
            #add the above dict to the overall dict   
            large_dict[folder + "/" + filename] = d_words
            file.close()
    return large_dict


# ### 1.2.3: TF-IDF Model
# 
# The first part of vocabulary extraction is the same as above.
# 
# Given the Document collection $\mathcal{D}$, calculate the Inverse Document Frequency (IDF) for each word in the vocabulary $\mathcal{V}$. The IDF of the word $v$ is defined as the log (use base 10) of the multiplicative inverse of the fraction of documents in $\mathcal{D}$ that contain $v$. That is:
# $$idf(v) = \log\frac{|\mathcal{D}|}{|\{d \in \mathcal{D} ;v \in d\}|}$$
# 
# Similar to the representation above, you will represent each document $d \in \mathcal{D}$ as a vector of all the tokens in $\mathcal{V}$ that appear in $d$, along with their <b>tf idf</b> value. Specifically, you can think of representing the collection as a matrix $f[d,v]$, defined as follows:
# \begin{equation*}
# f[d, v] = tf(d, v) \cdot idf(v, \mathcal{D}), ~~~~\forall v \in  \mathcal{V}, ~~~~\forall d \in  \mathcal{D},
# \end{equation*}
# 
# where, $tf(.)$ is the <i>Term-Frequency</i>, and $idf(.)$ is the <i>Inverse Document-Frequency</i> as defined above.

# In[284]:


def tf_idf(directory_name, use_adv_tokenization=False, useStop = True):
    '''
    Construct a dictionary mapping document names to dictionaries of words to the 
    its TF-IDF value, with respect to the document
    
    :param directory_name: name of the directory where the train dataset is stored
    :param use_adv_tokenization: if False then use simple space separation,
                                 else use a more advanced tokenization scheme
    :return: TF-IDF Model
    '''
    large_dict = {}
    sw = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    #go through each folder in the directory and then each doc
    for folder in os.listdir(directory_name):
        for filename in os.listdir(directory_name + "/" + folder):
            docname = directory_name + "/" + folder + "/" + filename
            file = open(docname, 'r', errors='ignore')
            f_strings = file.read()
            tokenized_words = f_strings.split()
            #filter the bad words out
            if useStop:
                filtered_words = [lemmatizer.lemmatize(word.lower()) 
                                  for word in tokenized_words 
                                  if lemmatizer.lemmatize(word.lower()) not in sw]
            else:
                filtered_words = tokenized_words
            #create a new dictionary that will map words to freqs
            d_words = {}
            for word in filtered_words:
                if word in d_words:
                    d_words[word] =  d_words[word]  + 1
                else:
                    d_words[word] = 1
            #the big dictionary is then added to with the given doc and the above new dictionary
            large_dict[folder + "/" + filename] = d_words
            file.close()
    
    totalDocs = len(large_dict.keys())
    #dict that maps from word to number of docs appeared
    #get the term frequency counts for tf-idf
    new_dict = {}
    for doc in large_dict.keys():
        for word in large_dict[doc]:
            if word in new_dict:
                new_dict[word] = new_dict[word] + 1
            else:
                new_dict[word] = 1;
            
    
    for doc in large_dict.keys():
        for word in large_dict[doc]:
            #find the td-idf score for each word
            large_dict[doc][word] = large_dict[doc][word] * math.log10(totalDocs/ new_dict[word])
    
    
    
    
    return large_dict


# ## 1.3: Experiment 1
# 
# In this experiment, you will implement a simple (multiclass) Naive Bayes Classifier. That is, you will use the training documents to learn a model and then compute the most likely label among the 20 labels for a new document, using the Naive Bayes assumption. You will do it for all three document representations defined earlier. 
# 
# Note that, using Bayes rule, your prediction should be:
# 
# \begin{equation*}
# \hat{y}_d = \underset{y}{\mathrm{argmax}} P(d | y) \cdot P(y),
# \end{equation*}
# where $y$ ranges over the $20$ candidate labels.
# 
# Since we are using the Naive Bayes model, that is, we assume the independence of the features (words in a document) given the label (document type), we get: 
# 
# \begin{equation*}
# P(d | y) = \Pi_{v \in d}  P(v | y)
# \end{equation*}
# where $v$ is a word in document $d$ and $y$ is the label of document $d$. 
# 
# The question is now how to estimate the coordinate-wise conditional probabilities $P(v|y)$. We have suggested to use three different representations of the document, but in all cases, estimate this conditional probability as: 
# \begin{equation*}
# P(v | y) = \frac{\sum_{\text{docs $d' \in \mathcal{D}$ of class $y$}}f[d',v]}{\sum_{w \in \mathcal{V}}\sum_{\text{docs $d' \in \mathcal{D}$ of class $y$}}f[d',w]}
# \end{equation*}
# Notice that in class we used the same estimation for the <b>B-BOW</b> model, and here we generalize it to the other two representations. 
# 
# To do <b>$k$-laplace smoothing</b> (choose a suitable value of $k$), modify the above formula as follows: (You can think of 1-laplace smoothing as adding another document of class $y$ which contains all the words in the vocabulary.)
# 
# \begin{equation*}
# P(v | y) = \frac{\sum_{\text{docs $d' \in \mathcal{D}$ of class $y$}}f[d',v] + k}{(\sum_{w \in \mathcal{V}}\sum_{\text{docs $d' \in \mathcal{D}$ of class $y$}}f[d',w]) + |\mathcal{V}|k}.
# \end{equation*}
# You will also need to estimate the prior probability of each label, as:
# \begin{equation*}
# P(y) = \frac{\text{# of documents in }\mathcal{D} \text{ with label $y$}}{|\mathcal{D}|}
# \end{equation*}
# 
# In this experiment you will run the Naive Bayes algorithm for both datasets and all three the representations and provide an analysis of your results. Use <b>accuracy</b> to measure the performance.
# 
# </b>Important Implementation Detail:</b> Since we are multiplying probabilities, numerical underflow may occur while computing the resultant values. To avoid this, you should do your computation in <b>log space</b>. This means that you should take the log of the probability.
# <b>Also feel free to add in your own optional parameters for optimization purposes!</b>

# In[285]:


def precalculate(model, k):
    rel_dirs = {}
    total_count = 0
    other_dirs = {}
    denominators = {}
    vSet = set()

    #go through each doc
    for doc in model:
        #update total count of docs
        total_count += 1
        classa = doc.split("/")[0]
        
        #update other_dirs and set the germane counts
        if classa in rel_dirs.keys():
            rel_dirs[classa] += 1
            other_dirs[classa].append(doc)
        else:
            rel_dirs[classa] = 1
            other_dirs[classa] = []
            other_dirs[classa].append(doc)
    
    #compute the denominator
    numerators = {classa: {} for classa in other_dirs}
    for doc in model:
        classa = doc.split("/")[0]
        for word in model[doc]:
            #add the word to our overall vocabulary
            vSet.add(word)
            if word in numerators[classa]:
                numerators[classa][word] += model[doc][word]
            else:
                numerators[classa][word] = model[doc][word] + k
            
    #loop through each label and calculate the denominators for ech label
    for classa in other_dirs.keys():
        denominator = len(vSet) * k
        for word in vSet:
            for doc in other_dirs[classa]:
                if word in model[doc].keys():                   
                    denominator += model[doc][word]
        denominators[classa] = denominator
        
    return rel_dirs, total_count, other_dirs, vSet, len(vSet), numerators, denominators 
    
    





def naive_bayes(document, train_dir, model, k=1, precalc = False, given_sizes = {}, total_count = 0, rel_dirs = {}, vSet = set(), v_set_size = 0, numerators = {}, denominators = {}, useStop =True):
    '''
    Uses naive bayes to predict the label of the given document.
    
    :param document: the path to the document whose label you need to predict
    :param train_dir: the directory with the training data
    :param model: the data representation model
    :param k: the k parameter in k-laplace smoothing
    :return: a tuple containing the predicted label of the document and log-probability of that label
    '''
    #don't use precalc values by default
    if not precalc:
        
        #precalc and get stopwords
        given_sizes, total_count, rel_dirs, vSet, v_set_size, numerators, denominators  = precalculate(model, k)  
        stopwords1 = stopwords.words('english')

        #get the germane words
        fia = open(document, 'r', errors='ignore')
        splitter = fia.read()
        tokenized_words = splitter.split()
        if useStop:
            filtered_words = [WordNetLemmatizer().lemmatize(word.lower()) 
                                      for word in tokenized_words 
                                      if WordNetLemmatizer().lemmatize(word.lower()) not in stopwords1]
        else:
            filtered_words = tokenized_words
        fia.close()

        prdy = {}
        #go through each label
        for classa in rel_dirs:
            pry = 0
            #Get Pr v given y for each given v in d
            for word in filtered_words:    
                if classa in numerators and word in numerators[classa]:
                    pry += np.log((numerators[classa][word] * 1.0) / (1.0 * denominators[classa]))
                else:
                    pry += np.log((1.0) / (v_set_size))

            prdy[classa] = pry  
   


        #Find the log of all prob d given y * prob y and store into dictionary
        bvs = {}
        for label in given_sizes.keys():
            bvs[label] = prdy[label] + np.log(given_sizes[label] * 1.0 / total_count)
        return (max(bvs, key=bvs.get), bvs[label])
    else:
        
        #define stopwords and get filtered words
        stopwords1 = stopwords.words('english')
        fia = open(document, 'r', errors='ignore')
        splitter = fia.read()
        tokenized_words = splitter.split()
        filtered_words = [WordNetLemmatizer().lemmatize(word.lower()) 
                                  for word in tokenized_words 
                                  if WordNetLemmatizer().lemmatize(word.lower()) not in stopwords1]
        fia.close()

        prdy = {}
        #go through each label
        for classa in rel_dirs:
            pry = 0
            #Get Pr v given y for each given v in d
            for word in filtered_words:    
                if classa in numerators and word in numerators[classa]:
                    pry += np.log((numerators[classa][word] * 1.0) / (1.0 * denominators[classa]))
                else:
                    pry += np.log((1.0) / (v_set_size))

            prdy[classa] = pry  

        #Find the log of all prob d given y * prob y and store into dictionary 
        bvs = {}
        for label in given_sizes.keys():
            bvs[label] = prdy[label] + np.log(given_sizes[label] * 1.0 / total_count)
        return (max(bvs, key=bvs.get), bvs[label])




# In[286]:


#Experiment 1 - with stop

# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-bydate-rm-metadata/test"
# train_dir = "20news-data/20news-bydate-rm-metadata/train"
# train_model= b_bow(train_dir)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("B_BOW accuracy: " + str(acc))


# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-bydate-rm-metadata/test"
# train_dir = "20news-data/20news-bydate-rm-metadata/train"
# train_model= c_bow(train_dir)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("C_BOW accuracy: " +  str(acc))


# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-bydate-rm-metadata/test"
# train_dir = "20news-data/20news-bydate-rm-metadata/train"
# train_model= tf_idf(train_dir)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("TF_IDF accuracy: " +  str(acc))





# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-random-rm-metadata/test"
# train_dir = "20news-data/20news-random-rm-metadata/train"
# train_model= b_bow(train_dir)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("B_BOW accuracy: " +  str(acc))


# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-random-rm-metadata/test"
# train_dir = "20news-data/20news-random-rm-metadata/train"
# train_model= c_bow(train_dir)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("C_BOW accuracy: " +  str(acc))


# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-random-rm-metadata/test"
# train_dir = "20news-data/20news-random-rm-metadata/train"
# train_model= tf_idf(train_dir)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("TF_IDF accuracy: " +  str(acc))



















#Experiment 1 - without stop

# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-bydate-rm-metadata/test"
# train_dir = "20news-data/20news-bydate-rm-metadata/train"
# train_model= b_bow(train_dir, use_adv_tokenization = False, useStop = False)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators, useStop = False)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("B_BOW accuracy: " + str(acc))


# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-bydate-rm-metadata/test"
# train_dir = "20news-data/20news-bydate-rm-metadata/train"
# train_model= c_bow(train_dir, use_adv_tokenization = False, useStop = False)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators, useStop = False)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("C_BOW accuracy: " +  str(acc))


# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-bydate-rm-metadata/test"
# train_dir = "20news-data/20news-bydate-rm-metadata/train"
# train_model= tf_idf(train_dir, use_adv_tokenization = False, useStop = False)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators, useStop = False)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("TF_IDF accuracy: " +  str(acc))





# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-random-rm-metadata/test"
# train_dir = "20news-data/20news-random-rm-metadata/train"
# train_model= b_bow(train_dir, use_adv_tokenization = False, useStop = False)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators, useStop = False)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("B_BOW accuracy: " +  str(acc))


# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-random-rm-metadata/test"
# train_dir = "20news-data/20news-random-rm-metadata/train"
# train_model= c_bow(train_dir, use_adv_tokenization = False, useStop = False)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators, useStop = False)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("C_BOW accuracy: " +  str(acc))


# total = 0
# total_correct = 0
# test_dir = "20news-data/20news-random-rm-metadata/test"
# train_dir = "20news-data/20news-random-rm-metadata/train"
# train_model= tf_idf(train_dir, use_adv_tokenization = False, useStop = False)
# y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)

# for folder in os.listdir(test_dir):
#     for filename in os.listdir(test_dir + "/" + folder):
#         total += 1
#         result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
#                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
#                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators, useStop = False)[0]
#         if folder == result:
#             total_correct += 1
# acc = total_correct / total

# print("TF_IDF accuracy: " +  str(acc))






# ## 1.4: Experiment 2
# 
# In this experiment, you will use Semi-Supervised Learning to do document classification. 
# The underlying algorithm you will use is Naive Bayes with the <b>B-BOW</b> representation, as defined in Experiment 1.
# 
# See write-up for more details.

# ### 1.4.1: Top-m
# 
# Filter the top-m instances and augment them to the labeled data.

# In[333]:


def filter_by_mtop(predictions, m=1000):
    '''
    Filter the top-m instances and augment them to the labeled data.
    
    :param predictions: dictionary mapping documents to tuples of (predicted label, log-probability)
                        i.e. {'doc1':('label1', prob1), 'doc2':('label2', prob2)}
    :param k: the number of predictions to augment to the labeled data
    :return: a tuple of (dictionary mapping documents to labels of new supervised data,
                         list of documents in unsupervised data)
    '''
    #list1 has the prob to label to doc
    list1 = []
    for doc in predictions.keys(): 
        list1.append((predictions[doc][1], predictions[doc][0], doc))
    #use a heap to get the best-m results and get the requisite format
    new_dict = {}
    for i in heapq.nlargest(m,list1):
        new_dict[i[2]] = i[1]
    #make the relevant list
    noList = []
    for doc in predictions.keys(): 
        if doc not in new_dict.keys():
            noList.append(doc)
    return(new_dict, noList)

    
            



# ### 1.4.2: Threshold
# 
# Set a threshold. Augment those instances to the labeled data with confidence strictly higher than this threshold. You have to be careful here to make sure that the program terminates. If the confidence is never higher than the threshold, then the procedure will take forever to terminate. You can choose to terminate the program if there are 5 consecutive iterations where no confidence exceeded the threshold value. 

# In[334]:


def filter_by_threshold(predictions, threshold=-300):
    '''
    Augment instances to the labeled data with confidence strictly higher than the given threshold.
    
    :param predictions: dictionary mapping documents to tuples of (predicted label, log-probability)
                        i.e. {'doc1':('label1', prob1), 'doc2':('label2', prob2)}
    :param threshold: the threshold to filter by
    :return: a tuple of (dictionary mapping documents to labels of new supervised data,
                         list of documents in unsupervised data)
    '''
    #list1 has the prob to label to doc
    list1 = []
    for doc in predictions.keys(): 
        list1.append((predictions[doc][1], predictions[doc][0], doc))
    
    #loop through until no change after 5 iterations
    #and get values above the threshold
    new_dict = {}
    counter = 0
    for i in list1:
        if counter == 5:
            break;
        if(i[0] <= threshold):
            counter+=1
        else:
            new_dict[i[2]] = i[1]

            
            
        
    #get the format back to the way we wanted it
    noList = []
    for doc in predictions.keys(): 
        if doc not in new_dict.keys():
            noList.append(doc)
    return(new_dict, noList)


# In[347]:


import random
from math import floor

def semi_supervised(test_dir, train_dir, filter_function, p=0.05, isTesting = False):
    '''
    Semi-supervised classifier with Naive Bayes and B-BoW representation.
    
    :param test_dir: directory with the test data
    :param train_dir: directory with the train data
    :param filter_function: the function we will use to filter data for augmenting the labeled data
    :param p: the proportion of the training data that starts off as labeled
    :return: a tuple containing the model trained on the supervised data, and the supervised dataset
             where the model is represented by a dict (same as B-BoW),
             and S is a mapping from documents to labels
    '''
    
    #init our bbow model
    train_bbow = b_bow(train_dir, False)
            
    #init the supervised and unsupervised doc lists
    sdocs = random.sample(train_bbow.keys(), floor(p *  len(train_bbow)))
    undocs = [doc for doc in train_bbow.keys() if doc not in sdocs]

    train_model = {doc: train_bbow[doc] for doc in sdocs}    
    doct = {}
    
    for doc in sdocs:
        label = doc.split('/')[0]
        doct[doc] = label
    
    
    #loop until we have >= 5 unchanged supervised sets
    counter = 0    
    while counter < 5:   

        y_sizes, doc_count, y_file_dirs, vocabulary, size_of_vocabulary, numerators, denominators = precalculate(train_model, 1)
        
        #if no supervised then exit
        if len(undocs) == 0:
            return train_model, doct
        naive_bayes_results = {doc: naive_bayes(train_dir + '/' + doc, train_dir, train_model, 1, True,
                                               given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
                                               v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators)
                               for doc in undocs}        
        new_dit,  nothing = filter_function(naive_bayes_results)
        
        #calculate accuracies for testing
        if isTesting:      
            total = 0
            total_correct = 0  
            #go through each folder and doc and get the NB result
            for folder in os.listdir(test_dir):
                for filename in os.listdir(test_dir + "/" + folder):
                    result = naive_bayes(test_dir + '/' + folder + '/' + filename, train_dir, train_model, 1, True,
                                                given_sizes = y_sizes, total_count = doc_count, rel_dirs = y_file_dirs, vSet = vocabulary, 
                                                v_set_size = size_of_vocabulary, numerators = numerators, denominators = denominators, useStop = False)[0]
                    total += 1 
                    #if equal, then increment correct
                    if folder ==  result:
                        total_correct += 1
            acc = total_correct / total
            print(acc)

        #update the counter if no change
        if len(new_dit) == 0:
            counter += 1
        else:    
            for doc in new_dit.keys():
                train_model[doc] = train_bbow[doc]
                doct[doc] = new_dit[doc][0]
            for doc in new_dit.keys():
                undocs.remove(doc)
    return train_model, doct
  


  


# In[348]:


#Experiment 2- by date

# semi_supervised('20news-data/20news-bydate-rm-metadata/test', '20news-data/20news-bydate-rm-metadata/train', filter_by_threshold, p=0.05, isTesting = True)
# print('')
# semi_supervised('20news-data/20news-bydate-rm-metadata/test', '20news-data/20news-bydate-rm-metadata/train', filter_by_threshold, p=0.10, isTesting = True)
# print('')
# semi_supervised('20news-data/20news-bydate-rm-metadata/test', '20news-data/20news-bydate-rm-metadata/train', filter_by_threshold, p=0.50, isTesting = True)


# print('')


# semi_supervised('20news-data/20news-bydate-rm-metadata/test', '20news-data/20news-bydate-rm-metadata/train', filter_by_mtop, p=0.05, isTesting = True)
# print('')
# semi_supervised('20news-data/20news-bydate-rm-metadata/test', '20news-data/20news-bydate-rm-metadata/train', filter_by_mtop, p=0.10, isTesting = True)
# print('')

# semi_supervised('20news-data/20news-bydate-rm-metadata/test', '20news-data/20news-bydate-rm-metadata/train', filter_by_mtop, p=0.50, isTesting = True)




# #Experiment 2- random

# print('')

# semi_supervised('20news-data/20news-random-rm-metadata/test', '20news-data/20news-random-rm-metadata/train', filter_by_threshold, p=0.05, isTesting = True)
# print('')

# semi_supervised('20news-data/20news-random-rm-metadata/test', '20news-data/20news-random-rm-metadata/train', filter_by_threshold, p=0.10, isTesting = True)
# print('')

# semi_supervised('20news-data/20news-random-rm-metadata/test', '20news-data/20news-random-rm-metadata/train', filter_by_threshold, p=0.50, isTesting = True)

# print('')

# semi_supervised('20news-data/20news-random-rm-metadata/test', '20news-data/20news-random-rm-metadata/train', filter_by_mtop, p=0.05, isTesting = True)
# print('')

# semi_supervised('20news-data/20news-random-rm-metadata/test', '20news-data/20news-random-rm-metadata/train', filter_by_mtop, p=0.10, isTesting = True)
# print('')

# semi_supervised('20news-data/20news-random-rm-metadata/test', '20news-data/20news-random-rm-metadata/train', filter_by_mtop, p=0.50, isTesting = True)



# ## 1.5: (Optional: Extra Credit) Experiment 3
# 
# For experiment 2, initialize as suggested but, instead of the filtering, run EM. That is, for each data point in U, label it fractionally -- label data point d with label $l$, that has weight $p(l|d)$. Then, add all the (weighted) examples to S. Now use Naive Bayes to learn again on the augmented data set (but now each data point has a weight! That is, when you compute $P(f[d,i]|y)$, rather than simply counting all the documents that have label $y$, now <b>all</b> the documents have "fractions" of the label $y$), and use the model to relabel it (again, fractionally, as defined above.) Iterate, and determine a stopping criteria. 

# In[51]:


# Experiment 3


# ## Running your experiments
# 
# Use the cell below to run your code.
# 
# 
# Remember to comment it all out before submitting!
