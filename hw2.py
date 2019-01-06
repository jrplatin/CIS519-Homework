
# coding: utf-8

# In[182]:


import sklearn
import numpy 
import random
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer


# In[91]:


def calc_f1(true_labels,predicted_labels):
    # true_labels - list of true labels (1/-1)
    # predicted_labels - list of predicted labels (1/-1)
    # return precision, recall and f1
    labelledPos = 0
    correctPos = 0
    actualPos = 0
  
    
    for i in range(len(true_labels)):    
        if true_labels[i] == 1:
            actualPos+=1
        if predicted_labels[i] == 1:
            labelledPos+=1
        if predicted_labels[i] == true_labels[i] and predicted_labels[i] == 1:
            correctPos+=1
    precision = float(correctPos / labelledPos)
    recall = float(correctPos / actualPos)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


# In[92]:


from sklearn.svm import LinearSVC
class Classifier(object):
    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged = False, eta = 1, alpha = 1):
        # Algorithm values can be Perceptron, Winnow, Adagrad, Perceptron-Avg, Winnow-Avg, Adagrad-Avg, SVM
        # Get features from examples; this line figures out what features are present in
        # the training data, such as 'w-1=dog' or 'w+1=cat'
        self.features = {feature for xi in x_train for feature in xi.keys()}
        
        if algorithm == 'Perceptron':
            #Initialize w, bias
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] + eta*yi*value
                        self.w['bias'] = self.w['bias'] + eta*yi

        elif algorithm == 'Winnow':
              #Initialize w, bias
            self.w, self.w['bias'] = {feature:1.0 for feature in self.features}, -len(self.features)
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] * alpha**(yi*value)
        
        elif algorithm == 'Adagrad':
             #Initialize w, bias, gradient accumulator
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            BigG, BigG['bias'] = {feature:0.0 for feature in self.features}, 0.0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.dotproduct(xi) * yi
                    #Update weights if there is a misclassification
                    if y_hat <= 1:
                        for feature, value in xi.items():
                            littleg = -yi * value
                            BigG[feature] = BigG[feature] + pow(littleg, 2)
                            self.w[feature] = self.w[feature] - eta* littleg / math.sqrt(BigG[feature])
                        BigG['bias'] = BigG['bias'] + pow(-yi, 2)
                        self.w['bias'] = self.w['bias'] + eta * yi / math.sqrt(BigG['bias'])
                              
                             
                        
        elif algorithm == 'Perceptron-Avg':            
            #Initialize w, bias
            self.w, self.w['bias'], c, c_total = {feature:0.0 for feature in self.features}, 0.0, 0.0, 0.0
            w_acc = (self.w).copy()
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        #think about merging below and 2 below not all features, only through xi.items()
                        #group in terms of delta instead of C, 
                        for feature, value in xi.items():
                            self.w[feature] += eta*yi*value  #delta is latter
                            w_acc[feature] += c_total * eta * yi * value
                        self.w['bias'] = self.w['bias'] + eta*yi
                        w_acc['bias'] += c_total * eta * yi
                        c_total += c
                        c = 1
                    else:
                        c+=1
     
            for feature in self.features:
                self.w[feature] -= w_acc[feature] / c_total
            self.w['bias'] -= w_acc['bias'] / c_total
        
        elif algorithm == 'Winnow-Avg':
            #Initialize w, bias
            self.w, self.w['bias'] = {feature:1.0 for feature in self.features}, -len(self.features)
            w_acc, w_acc['bias'] = {feature:1.0 for feature in self.features}, -len(self.features)
            c = 0
            c_total = 0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.predict(xi)
                    #Update weights if there is a misclassification
                    if yi != y_hat:
                        for feature in self.features:
                            w_acc[feature] = w_acc[feature] + c * self.w[feature]
                        for feature, value in xi.items():
                            self.w[feature] = self.w[feature] * alpha**(yi*value)
                        c_total += c
                        c = 1
                    else:
                        c+=1
            c_total += c
            for feature in self.features:
                w_acc[feature] = w_acc[feature] + c * self.w[feature]
                w_acc[feature] /= c_total
            self.w = w_acc
       
        elif algorithm == 'Adagrad-Avg':
              #Initialize w, bias, gradient accumulator
            self.w, self.w['bias'] = {feature:0.0 for feature in self.features}, 0.0
            w_acc, w_acc['bias'] = {feature:0.0 for feature in self.features}, 0.0
            BigG, BigG['bias'] = {feature:0.0 for feature in self.features}, 0.0
            c = 0
            c_total = 0
            #Iterate over the training data n times
            for j in range(iterations):
                #Check each training example
                for i in range(len(x_train)):
                    xi, yi = x_train[i], y_train[i]
                    y_hat = self.dotproduct(xi) * yi
                    #Update weights if there is a misclassification
                    if y_hat <= 1:
                        for feature in self.features:
                            w_acc[feature] = w_acc[feature] + c * self.w[feature]
                        for feature, value in xi.items():
                            g = -yi * value
                            BigG[feature] = BigG[feature] + pow(g, 2)
                            self.w[feature] = self.w[feature] - eta* g / math.sqrt(BigG[feature])
                        BigG['bias'] = BigG['bias'] + pow(-yi, 2)
                        w_acc['bias'] = w_acc['bias'] + c * self.w['bias']
                        self.w['bias'] = self.w['bias'] + eta * yi / math.sqrt(BigG['bias'])
                        c_total += c
                        c = 1
                    else:
                        c+=1
            c_total += c
            for feature in self.features:
                w_acc[feature] = w_acc[feature] + c * self.w[feature]
                w_acc[feature] /= c_total
            w_acc['bias'] = w_acc['bias'] + c * self.w['bias']
            w_acc['bias'] /= c_total
            self.w = w_acc
           
        
        elif algorithm == 'SVM':
            self.svm = LinearSVC(penalty='l2', loss='hinge')
            self.vectorizer = DictVectorizer()
            x_train = self.vectorizer.fit_transform(x_train)
            self.svm.fit(x_train, y_train)
        
        
        else:
            print('Unknown algorithm')
                
    def predict(self, x):
        s = sum([self.w[feature]*value for feature, value in x.items()]) + self.w['bias']
        return 1 if s > 0 else -1
    
    def dotproduct(self, x):
        return sum([self.w[feature]*value for feature, value in x.items()]) + self.w['bias']
    
    
    def predict_SVM(self, x):
        x = self.vectorizer.transform([x])
        return self.svm.predict(x)[0]


# In[93]:


#Parse the real-world data to generate features, 
#Returns a list of tuple lists
def parse_real_data(path):
    #List of tuples for each sentence
    data = []
    for filename in os.listdir(path):
        with open(path+filename, 'r') as file:
            sentence = []
            for line in file:
                if line == '\n':
                    data.append(sentence)
                    sentence = []
                else:
                    sentence.append(tuple(line.split()))
    return data


# In[94]:


#Returns a list of labels
def parse_synthetic_labels(path):
    #List of tuples for each sentence
    labels = []
    with open(path+'y.txt', 'rb') as file:
        for line in file:
            labels.append(int(line.strip()))
    return labels


# In[95]:


#Returns a list of features
def parse_synthetic_data(path):
    #List of tuples for each sentence
    data = []
    with open(path+'x.txt') as file:
        features = []
        for line in file:
            #print('Line:', line)
            for ch in line:
                if ch == '[' or ch.isspace():
                    continue
                elif ch == ']':
                    data.append(features)
                    features = []
                else:
                    features.append(int(ch))
    return data


# In[96]:


def extract_features_train(news_train_data):
    news_train_y = []
    news_train_x = []
    train_features = set([])
    for sentence in news_train_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3,len(padded)-3):
            news_train_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-3='+str(padded[i-3][0])
            feat2 = 'w-2='+str(padded[i-2][0])
            feat3 = 'w-1='+str(padded[i-1][0])
            feat4 = 'w+1='+str(padded[i+1][0])
            feat5 = 'w+2='+str(padded[i+2][0])
            feat6 = 'w+3='+str(padded[i+3][0])
            feat7 = 'w-1&w-2='+str(padded[i-1][0])+ ' ' + str(padded[i-2][0])
            feat8 = 'w+1&w+2='+str(padded[i+1][0])+  ' ' +  str(padded[i+2][0])
            feat9 = 'w-1&w+1='+str(padded[i-1][0])+  ' ' +  str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            train_features.update(feats)
            feats = {feature:1 for feature in feats}
            news_train_x.append(feats)
    return train_features, news_train_x, news_train_y


# In[97]:


def extract_features_dev(news_dev_data, train_features):
    news_dev_y = []
    news_dev_x = []
    for sentence in news_dev_data:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3,len(padded)-3):
            news_dev_y.append(1 if padded[i][1]=='I' else -1)
            feat1 = 'w-3='+str(padded[i-3][0])
            feat2 = 'w-2='+str(padded[i-2][0])
            feat3 = 'w-1='+str(padded[i-1][0])
            feat4 = 'w+1='+str(padded[i+1][0])
            feat5 = 'w+2='+str(padded[i+2][0])
            feat6 = 'w+3='+str(padded[i+3][0])
            feat7 = 'w-1&w-2='+str(padded[i-1][0])+ ' ' +str(padded[i-2][0])
            feat8 = 'w+1&w+2='+str(padded[i+1][0])+ ' ' + str(padded[i+2][0])
            feat9 = 'w-1&w+1='+str(padded[i-1][0])+ ' ' + str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            feats = {feature:1 for feature in feats if feature in train_features}
            news_dev_x.append(feats)
    return news_dev_x, news_dev_y


# In[98]:


#extract_features for the test data
def extract_features_test(train_data1, train_features):
    news_dev_x = []
    for sentence in train_data1:
        padded = sentence[:]
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.insert(0, ('SSS', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        padded.append(('EEE', None))
        for i in range(3,len(padded)-3):
            feat1 = 'w-3='+str(padded[i-3][0])
            feat2 = 'w-2='+str(padded[i-2][0])
            feat3 = 'w-1='+str(padded[i-1][0])
            feat4 = 'w+1='+str(padded[i+1][0])
            feat5 = 'w+2='+str(padded[i+2][0])
            feat6 = 'w+3='+str(padded[i+3][0])
            feat7 = 'w-1&w-2='+str(padded[i-1][0])+ ' ' +str(padded[i-2][0])
            feat8 = 'w+1&w+2='+str(padded[i+1][0])+ ' ' + str(padded[i+2][0])
            feat9 = 'w-1&w+1='+str(padded[i-1][0])+ ' ' + str(padded[i+1][0])
            feats = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]
            feats = {feature:1 for feature in feats if feature in train_features}
            news_dev_x.append(feats)
    return news_dev_x


# In[99]:


# print('Loading data...')
#  #Load data from folders.
#  #Real world data - lists of tuple lists
# news_train_data = parse_real_data('Data/Real-World/CoNLL/Train/')
# news_dev_data = parse_real_data('Data/Real-World/CoNLL/Dev/')
# news_test_data = parse_real_data('Data/Real-World/CoNLL/Test/')
# email_dev_data = parse_real_data('Data/Real-World/Enron/Dev/')
# email_test_data = parse_real_data('Data/Real-World/Enron/Test/')

# #Load dense synthetic data
# syn_dense_train_data = parse_synthetic_data('Data/Synthetic/Dense/Train/')
# syn_dense_train_labels = parse_synthetic_labels('Data/Synthetic/Dense/Train/')
# syn_dense_dev_data = parse_synthetic_data('Data/Synthetic/Dense/Dev/')
# syn_dense_dev_labels = parse_synthetic_labels('Data/Synthetic/Dense/Dev/')
# syn_dense_dev_no_noise_data = parse_synthetic_data('Data/Synthetic/Dense/Dev_no_noise/')
# syn_dense_dev_no_noise_labels = parse_synthetic_labels('Data/Synthetic/Dense/Dev_no_noise/')

# syn_dense_test_data = parse_synthetic_data('Data/Synthetic/Dense/Test/')
# syn_sparse_test_data= parse_synthetic_data('Data/Synthetic/Sparse/Test/')
   
# #Load sparse synthetic data
# syn_sparse_train_data = parse_synthetic_data('Data/Synthetic/Sparse/Train/')
# syn_sparse_train_labels = parse_synthetic_labels('Data/Synthetic/Sparse/Train/')
# syn_sparse_dev_data = parse_synthetic_data('Data/Synthetic/Sparse/Dev/')
# syn_sparse_dev_labels = parse_synthetic_labels('Data/Synthetic/Sparse/Dev/')

# #Load the no_noise data
# syn_dense_no_noise_data = parse_synthetic_data('Data/Synthetic/Dense/Dev_no_noise/')
# syn_dense_no_noise_labels = parse_synthetic_labels('Data/Synthetic/Dense/Dev_no_noise/')

# print('Data Loaded.')


# In[100]:


# # Convert to sparse dictionary representations.


# print('Converting Synthetic data...')
# syn_dense_train = zip(*[({'x'+str(i): syn_dense_train_data[j][i]
#     for i in range(len(syn_dense_train_data[j])) if syn_dense_train_data[j][i] == 1}, syn_dense_train_labels[j]) 
#         for j in range(len(syn_dense_train_data))])
# syn_dense_train_x, syn_dense_train_y = syn_dense_train

# syn_dense_dev = zip(*[({'x'+str(i): syn_dense_dev_data[j][i]
#     for i in range(len(syn_dense_dev_data[j])) if syn_dense_dev_data[j][i] == 1}, syn_dense_dev_labels[j]) 
#         for j in range(len(syn_dense_dev_data))])
# syn_dense_dev_x, syn_dense_dev_y = syn_dense_dev



# # Similarly add code for the dev set with no noise and sparse data
# syn_dense_dev_no_noise = zip(*[({'x'+str(i): syn_dense_dev_no_noise_data[j][i]
#     for i in range(len(syn_dense_dev_no_noise_data[j])) if syn_dense_dev_no_noise_data[j][i] == 1}, syn_dense_dev_no_noise_labels[j]) 
#         for j in range(len(syn_dense_dev_no_noise_data))])
# syn_dense_dev_no_noise_x, syn_dense_dev_no_noise_y = syn_dense_dev_no_noise




# syn_sparse_train = zip(*[({'x'+str(i): syn_sparse_train_data[j][i]
#     for i in range(len(syn_sparse_train_data[j])) if syn_sparse_train_data[j][i] == 1}, syn_sparse_train_labels[j]) 
#         for j in range(len(syn_sparse_train_data))])
# syn_sparse_train_x, syn_sparse_train_y = syn_sparse_train


# syn_sparse_dev = zip(*[({'x'+str(i): syn_sparse_dev_data[j][i]
#     for i in range(len(syn_sparse_dev_data[j])) if syn_sparse_dev_data[j][i] == 1}, syn_sparse_dev_labels[j]) 
#         for j in range(len(syn_sparse_dev_data))])
# syn_sparse_dev_x, syn_sparse_dev_y = syn_sparse_dev

# syn_sparse_test = zip(*[({'x'+str(i): syn_sparse_test_data[j][i]
#     for i in range(len(syn_sparse_test_data[j])) if syn_sparse_test_data[j][i] == 1}, syn_sparse_test_data[j]) 
#         for j in range(len(syn_sparse_test_data))])
# syn_sparse_test_x, syn_sparse_test_y = syn_sparse_test

# syn_dense_test = zip(*[({'x'+str(i): syn_dense_test_data[j][i]
#     for i in range(len(syn_dense_test_data[j])) if syn_dense_test_data[j][i] == 1}, syn_dense_test_data[j]) 
#         for j in range(len(syn_dense_test_data))])
# syn_dense_test_x, syn_dense_test_y = syn_dense_test


# syn_dense_no_noise = zip(*[({'x'+str(i): syn_dense_no_noise_data[j][i]
#     for i in range(len(syn_dense_test_data[j])) if syn_dense_no_noise_data[j][i] == 1}, syn_dense_no_noise_labels[j]) 
#         for j in range(len(syn_dense_test_data))])
# syn_dense_no_noise_x, syn_dense_no_noise_y = syn_dense_no_noise



# print('Done')




 
        


# In[101]:


# # Feature extraction
# # Remember to add the other features mentioned in the handout
# print('Extracting features from real-world data...')
# train_features, news_train_x, news_train_y = extract_features_train(news_train_data)
# news_dev_x, news_dev_y = extract_features_dev(news_dev_data, train_features)
# email_dev_x, email_dev_y = extract_features_dev(email_dev_data, train_features)

# email_test_data1 = extract_features_test(email_test_data, train_features)
# news_test_data1 = extract_features_test(news_test_data, train_features)


# print('Done')


# In[102]:


# #parameter tuning
# print('Winnow Accuracy')
# for alp in [1.1, 1.01, 1.005, 1.0005, 1.0001]:    
#     p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y, iterations=1, alpha=alp, eta=1)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     print('Winnow Syn Sparse Dev Accuracy with alpha = ' + str(alp) + ':', accuracy)
    
#     p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=1, alpha=alp, eta=1)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     print('Winnow Syn Dense Dev Accuracy with alpha = ' + str(alp) + ':', accuracy)
#     print('')
    
# print('Adagrad Accuracy')
# for eta in [1.5, 0.25, 0.03, 0.005, 0.001]:
#     # Test Perceptron on Dense Synthetic
#     p = Classifier('Adagrad', syn_sparse_train_x, syn_sparse_train_y, iterations=1, alpha=1, eta=eta)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     print('Adagrad Syn Sparse Dev Accuracy with eta = ' + str(eta) + ':', accuracy)
    
#     p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=1, alpha=1, eta=eta)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     print('Adagrad Syn Dense Dev Accuracy with eta = ' + str(eta) + ':', accuracy)
#     print('')


# In[189]:


# #Dense Plot (7 curves on one graph)
# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')   
# ax.set_xlim(0,5000)
# ax2.set_xlim(49500,50500)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()


# train_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 50000]
# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_dense_train_x[i])
#         randomY.append(syn_dense_train_y[i])
#     p = Classifier('SVM', randomX, randomY, iterations=10, alpha=1, eta=1)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict_SVM(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     train_acc.append(accuracy)  
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("SVM Accuracy " + str(train_acc[10]))

# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_dense_train_x[i])
#         randomY.append(syn_dense_train_y[i])
#     p = Classifier('Perceptron', randomX, randomY, iterations=10, alpha=1, eta=1)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     train_acc.append(accuracy)

# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Perceptron Accuracy " + str(train_acc[10]))


# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_dense_train_x[i])
#         randomY.append(syn_dense_train_y[i])
#     p = Classifier('Winnow', randomX, randomY, iterations=10, alpha=1.005, eta=1)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Winnow Accuracy " + str(train_acc[10]))





# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_dense_train_x[i])
#         randomY.append(syn_dense_train_y[i])
#     p = Classifier('Adagrad', randomX, randomY, iterations=10, alpha=1, eta=1.5)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Adagrad Accuracy " + str(train_acc[10]))





# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_dense_train_x[i])
#         randomY.append(syn_dense_train_y[i])
#     p = Classifier('Perceptron-Avg', randomX, randomY, iterations=10, alpha=1, eta=1)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Perceptron-Average Accuracy " + str(train_acc[10]))





# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_dense_train_x[i])
#         randomY.append(syn_dense_train_y[i])
#     p = Classifier('Winnow-Avg', randomX, randomY, iterations=10, alpha=1.005, eta=1)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Winnow-Average Accuracy " + str(train_acc[10]))




# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_dense_train_x[i])
#         randomY.append(syn_dense_train_y[i])
#     p = Classifier('Adagrad-Avg', randomX, randomY, iterations=10, alpha=1, eta=1.5)
#     accuracy = sum([1 for i in range(len(syn_dense_dev_y)) if p.predict(syn_dense_dev_x[i]) == syn_dense_dev_y[i]])/len(syn_dense_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Adagrad-Average Accuracy " + str(train_acc[10]))




# plt.legend(['SVM', 'Perceptron', 'Winnow', 'Adagrad', 'Perceptron-Avg', 'Winnow-Avg', 'Adagrad-Avg'], loc='lower right')
# plt.title("Dense Learning Curves")
# plt.xlabel("Training Size")
# plt.show()





# In[190]:


# #Sparse Plot (7 curves on one graph)

# f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')   
# ax.set_xlim(0,5000)
# ax2.set_xlim(49500,50500)
# ax.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# ax2.yaxis.tick_right()

# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_sparse_train_x[i])
#         randomY.append(syn_sparse_train_y[i])
#     p = Classifier('SVM', randomX, randomY, iterations=10, alpha=1, eta=1)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict_SVM(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     train_acc.append(accuracy)  
# print("SVM Accuracy " + str(train_acc[10]))
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 

# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_sparse_train_x[i])
#         randomY.append(syn_sparse_train_y[i])
#     p = Classifier('Perceptron', randomX, randomY, iterations=10, alpha=1, eta=1)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Perceptron Accuracy " + str(train_acc[10]))

# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_sparse_train_x[i])
#         randomY.append(syn_sparse_train_y[i])
#     p = Classifier('Winnow', randomX, randomY, iterations=10, alpha=1.005, eta=1)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Winnow Accuracy " + str(train_acc[10]))

# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_sparse_train_x[i])
#         randomY.append(syn_sparse_train_y[i])
#     p = Classifier('Adagrad', randomX[0:j], randomY[0:j], iterations=10, alpha=1, eta=1.5)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Adagrad Accuracy " + str(train_acc[10]))

# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_sparse_train_x[i])
#         randomY.append(syn_sparse_train_y[i])
#     p = Classifier('Perceptron-Avg', randomX, randomY, iterations=10, alpha=1, eta=1)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Perceptron-Average Accuracy " + str(train_acc[10]))

# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_sparse_train_x[i])
#         randomY.append(syn_sparse_train_y[i])
#     p = Classifier('Winnow-Avg', randomX, randomY, iterations=10, alpha=1.005, eta=1)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     train_acc.append(accuracy)
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
# print("Winnow-Average Accuracy " + str(train_acc[10]))



# train_acc = []
# for j in train_sizes:
#     randomX = []
#     randomY = []
#     for i in random.sample(range(0, j), j):
#         randomX.append(syn_sparse_train_x[i])
#         randomY.append(syn_sparse_train_y[i])
#     p = Classifier('Adagrad-Avg', randomX, randomY,  iterations=10, alpha=1, eta=1.5)
#     accuracy = sum([1 for i in range(len(syn_sparse_dev_y)) if p.predict(syn_sparse_dev_x[i]) == syn_sparse_dev_y[i]])/len(syn_sparse_dev_y)*100
#     train_acc.append(accuracy)
# print("Adagrad-Avg " + str(train_acc[10]))
# ax.plot(train_sizes, train_acc)
# ax2.plot(train_sizes, train_acc) 
    
# plt.legend(['SVM', 'Perceptron', 'Winnow', 'Adagrad', 'Perceptron-Avg', 'Winnow-Avg', 'Adagrad-Avg'], loc='lower right')
# plt.title("Sparse Learning Curves")
# plt.xlabel("Training Size")
# plt.show()


# In[105]:


# #report the precision, recall, and f1 for SVM and Perceptron-Avg on the news train and email dev/news dev

# p = Classifier('SVM', news_train_x, news_train_y,  iterations=10, alpha=1, eta=1.0)

# ap = sum([1 for i in range(len(news_train_y)) if news_train_y[i] == 1])
# lp = sum([1 for i in range(len(news_train_x)) if p.predict_SVM(news_train_x[i]) == 1])
# cp = sum([1 for i in range(len(news_train_x)) if p.predict_SVM(news_train_x[i]) == 1 and p.predict_SVM(news_train_x[i]) == news_train_y[i]])
# precision = cp / lp
# recall = cp / ap
# f1 = 2 * precision * recall / (precision + recall)
# print('SVM on news_train')
# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)
# print('')


# ap = sum([1 for i in range(len(news_dev_y)) if news_dev_y[i] == 1])
# lp = sum([1 for i in range(len(news_dev_x)) if p.predict_SVM(news_dev_x[i]) == 1])
# cp = sum([1 for i in range(len(news_dev_x)) if p.predict_SVM(news_dev_x[i]) == 1 and p.predict_SVM(news_dev_x[i]) == news_dev_y[i]])
# precision = cp / lp
# recall = cp / ap
# f1 = 2 * precision * recall / (precision + recall)
# print('SVM on news_dev')
# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)
# print('')


# ap = sum([1 for i in range(len(email_dev_y)) if email_dev_y[i] == 1])
# lp = sum([1 for i in range(len(email_dev_x)) if p.predict_SVM(email_dev_x[i]) == 1])
# cp = sum([1 for i in range(len(email_dev_x)) if p.predict_SVM(email_dev_x[i]) == 1 and p.predict_SVM(email_dev_x[i]) == email_dev_y[i]])
# precision = cp / lp
# recall = cp / ap
# f1 = 2 * precision * recall / (precision + recall)
# print('SVM on email_dev')
# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)
# print('')




# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, alpha=1, eta=1.5)


# ap = sum([1 for i in range(len(news_train_y)) if news_train_y[i] == 1])
# lp = sum([1 for i in range(len(news_train_x)) if p.predict(news_train_x[i]) == 1])
# cp = sum([1 for i in range(len(news_train_x)) if p.predict(news_train_x[i]) == 1 and p.predict(news_train_x[i]) == news_train_y[i]])
# precision = cp / lp
# recall = cp / ap
# f1 = 2 * precision * recall / (precision + recall)
# print('Perceptron-Avg on news_train')
# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)
# print('')


# ap = sum([1 for i in range(len(news_dev_y)) if news_dev_y[i] == 1])
# lp = sum([1 for i in range(len(news_dev_x)) if p.predict(news_dev_x[i]) == 1])
# cp = sum([1 for i in range(len(news_dev_x)) if p.predict(news_dev_x[i]) == 1 and p.predict(news_dev_x[i]) == news_dev_y[i]])
# precision = cp / lp
# recall = cp / ap
# f1 = 2 * precision * recall / (precision + recall)
# print('Perceptron-Avg on news_dev')
# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)
# print('')


# ap = sum([1 for i in range(len(email_dev_y)) if email_dev_y[i] == 1])
# lp = sum([1 for i in range(len(email_dev_x)) if p.predict(email_dev_x[i]) == 1])
# cp = sum([1 for i in range(len(email_dev_x)) if p.predict(email_dev_x[i]) == 1 and p.predict(email_dev_x[i]) == email_dev_y[i]])
# precision = cp / lp
# recall = cp / ap
# f1 = 2 * precision * recall / (precision + recall)
# print('Perceptron-Avg on email_dev')
# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)
# print('')








# In[106]:


# #training accuracy for dense and sparse, dev accuracy given above as last point in plot
#print('Training accuracy for dense')
# p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print("Perctron " + str(accuracy))

# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1.005, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print("Winnow " + str(accuracy))


# p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1.5)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print("Adagrad " + str(accuracy))


# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print("Perctron-Avg " + str(accuracy))


# p = Classifier('Winnow-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1.005, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print("Winnow-Avg " + str(accuracy))

# p = Classifier('Adagrad-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1.01, eta=1.5)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print("Adagrad-Avg " + str(accuracy))

# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_train_y)) if p.predict_SVM(syn_dense_train_x[i]) == syn_dense_train_y[i]])/len(syn_dense_train_y)*100
# print("SVM " + str(accuracy))

# print('')
#print('Training accuracy for sparse')
# #sparse
# p = Classifier('Perceptron', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha=1, eta=1)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print("Perctron " + str(accuracy))

# p = Classifier('Winnow', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha=1.005, eta=1)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print("Winnow " + str(accuracy))


# p = Classifier('Adagrad', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha=1, eta=1.5)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print("Adagrad " + str(accuracy))


# p = Classifier('Perceptron-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha=1, eta=1)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print("Perctron-Avg " + str(accuracy))


# p = Classifier('Winnow-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha=1.005, eta=1)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print("Winnow-Avg " + str(accuracy))

# p = Classifier('Adagrad-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha=1, eta=1.5)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print("Adagrad-Avg " + str(accuracy))

# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y, iterations=10, alpha=1.0, eta=1)
# accuracy = sum([1 for i in range(len(syn_sparse_train_y)) if p.predict_SVM(syn_sparse_train_x[i]) == syn_sparse_train_y[i]])/len(syn_sparse_train_y)*100
# print("SVM " + str(accuracy))



# In[107]:


# #no_noise accuracy
#print('No noise dev accuracy')
# p = Classifier('Perceptron', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_no_noise_y)) if p.predict(syn_dense_no_noise_x[i]) == syn_dense_no_noise_y[i]])/len(syn_dense_no_noise_y)*100
# print("Perctron " + str(accuracy))


# p = Classifier('Winnow', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1.005, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_no_noise_y)) if p.predict(syn_dense_no_noise_x[i]) == syn_dense_no_noise_y[i]])/len(syn_dense_no_noise_y)*100
# print("Winnow " +  str(accuracy))

# p = Classifier('Adagrad', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1.5)
# accuracy = sum([1 for i in range(len(syn_dense_no_noise_y)) if p.predict(syn_dense_no_noise_x[i]) == syn_dense_no_noise_y[i]])/len(syn_dense_no_noise_y)*100
# print("Adagrad " +  str(accuracy))

# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_no_noise_y)) if p.predict(syn_dense_no_noise_x[i]) == syn_dense_no_noise_y[i]])/len(syn_dense_no_noise_y)*100
# print("Perceptron-Avg " +  str(accuracy))

# p = Classifier('Winnow-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1.005, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_no_noise_y)) if p.predict(syn_dense_no_noise_x[i]) == syn_dense_no_noise_y[i]])/len(syn_dense_no_noise_y)*100
# print("Winnow-Avg " +  str(accuracy))

# p = Classifier('Adagrad-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1.5)
# accuracy = sum([1 for i in range(len(syn_dense_no_noise_y)) if p.predict(syn_dense_no_noise_x[i]) == syn_dense_no_noise_y[i]])/len(syn_dense_no_noise_y)*100
# print("Adagrad-Avg " +  str(accuracy))

# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y, iterations=10, alpha=1, eta=1)
# accuracy = sum([1 for i in range(len(syn_dense_no_noise_y)) if p.predict_SVM(syn_dense_no_noise_x[i]) == syn_dense_no_noise_y[i]])/len(syn_dense_no_noise_y)*100
# print("SVM " +  str(accuracy))


# In[108]:


# p = Classifier('Perceptron-Avg', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# with open ('p-sparse.txt', 'w') as file:
#     for i in range(len(syn_sparse_dev_x)):
#         file.write(str(p.predict(syn_sparse_test_x[i])) + '\n')

# p = Classifier('SVM', syn_sparse_train_x, syn_sparse_train_y, iterations=10)
# with open ('svm-sparse.txt', 'w') as file:
#     for i in range(len(syn_sparse_dev_x)):
#         file.write(str(p.predict_SVM(syn_sparse_test_x[i])) + '\n')

# p = Classifier('Perceptron-Avg', syn_dense_train_x, syn_dense_train_y, iterations=10, eta=1)
# with open ('p-dense.txt', 'w') as file:
#     for i in range(len(syn_dense_dev_x)):
#         file.write(str(p.predict(syn_dense_test_x[i])) + '\n')

# p = Classifier('SVM', syn_dense_train_x, syn_dense_train_y, iterations=10, eta=1)
# with open ('svm-dense.txt', 'w') as file:
#     for i in range(len(syn_dense_dev_x)):
#         file.write(str(p.predict_SVM(syn_dense_test_x[i])) + '\n')


# print('Done')


# In[109]:


# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# with open ('svm-enron.txt', 'w') as file:
#     for i in range(len(email_test_data1)):
#         file.write(('I' if p.predict_SVM(email_test_data1[i]) == 1 else 'O') + '\n')

# p = Classifier('SVM', news_train_x, news_train_y, iterations=10)
# with open ('svm-conll.txt', 'w') as file:
#     for i in range(len(news_test_data1)):
#          file.write(('I' if p.predict_SVM(news_test_data1[i]) == 1 else 'O') + '\n')
            
# print('svm done')


# print('started')
# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, eta=1)
# with open ('p-enron.txt', 'w') as file:
#     for i in range(len(email_test_data1)):
#         file.write(('I' if p.predict(email_test_data1[i]) == 1 else 'O') + '\n')
        
# print('other started')
# p = Classifier('Perceptron-Avg', news_train_x, news_train_y, iterations=10, eta=1)
# with open ('p-conll.txt', 'w') as file:
#     for i in range(len(news_test_data1)):
#          file.write(('I' if p.predict(news_test_data1[i]) == 1 else 'O') + '\n')
# print('avg done')

        



