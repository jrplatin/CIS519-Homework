
# coding: utf-8

# # Homework 1 Template
# This is the template for the first homework assignment.
# Below are some function templates which we require you to fill out.
# These will be tested by the autograder, so it is important to not edit the function definitions.
# The functions have python docstrings which should indicate what the input and output arguments are.

# ## Instructions for the Autograder
# When you submit your code to the autograder on Gradescope, you will need to comment out any code which is not an import statement or contained within a function definition.

# In[643]:


# Uncomment and run this code if you want to verify your `sklearn` installation.
# If this cell outputs 'array([1])', then it's installed correctly.

#from sklearn import linear_model
#X = [[0, 0], [1, 1]]
#y = [0, 1]
#clf = linear_model.SGDClassifier(loss = 'log')
#clf = clf.fit(X, y)
#clf.predict([[2, 2]])


# In[644]:


# Uncomment this code to see how to visualize a decision tree. This code should
# be commented out when you submit to the autograder.
# If this cell fails with
# an error related to `pydotplus`, try running `pip install pydotplus`
# from the command line, and retry. Similarly for any other package failure message.
# If you can't get this cell working, it's ok - this part is not required.
#
# This part should be commented out when you submit it to Gradescope

#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#import pydotplus
#
#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True,
#                feature_names=['feature1', 'feature2'],
#                class_names=['0', '1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())


# In[645]:


# This code should be commented out when you submit to the autograder.
# This cell will possibly download and unzip the dataset required for this assignment.
# It hasn't been tested on Windows, so it will not run if you are running on Windows.

#import os
#
#if os.name != 'nt':  # This is the Windows check
#    if not os.path.exists('badges.zip'):
#        # If your statement starts with "!", then the command is run in bash, not python
#        !wget https://www.seas.upenn.edu/~cis519/fall2018/assets/HW/HW1/badges.zip
#        !mkdir -p badges
#        !unzip badges.zip -d badges
#        print('The data has saved in the "badges" directory.')
#else:
#    print('Sorry, I think you are running on windows. '
#          'You will need to manually download the data')


# In[646]:


import numpy as np
def compute_features(name):
    """
    Compute all of the features for a given name. The input
    name will always have 3 names separated by a space.
    
    Args:
        name (str): The input name, like "bill henry gates".
    Returns:
        list: The features for the name as a list, like [0, 0, 1, 0, 1].a
    """
    
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    first, middle, last = name.split() 
    first = first[0:min(5, len(first))]
    middle = middle[0:min(5, len(first))]
    last = last[0:min(5, len(first))]
    
    relList = []
   
    
    for c in first:
        for l in alphabet:
            if(c == l):
                relList.append(1)
            else:
                relList.append(0)
    for c in range(0, 5- len(first)):
        for l in alphabet:
            relList.append(0)
            
            
    for c in middle:
        for l in alphabet:
            if(c == l):
                relList.append(1)
            else:
                relList.append(0)
    for c in range(0, 5- len(middle)):
        for l in alphabet:
            relList.append(0)
            
                
    for c in last:
        for l in alphabet:
            if(c == l):
                relList.append(1)
            else:
                relList.append(0)
    for c in range(0, 5- len(last)):
        for l in alphabet:
            relList.append(0)
    return relList


   


# In[647]:


from sklearn.tree import DecisionTreeClassifier

# The `max_depth=None` construction is how you specify default arguments
# in python. By adding a default argument, you can call this method in a couple of ways:
#     
#     train_decision_tree(X, y)
#     train_decision_tree(X, y, 4) or train_decision_tree(X, y, max_depth=4)
#
# In the first way, max_depth is automatically set to `None`, otherwise it is 4.
def train_decision_tree(X, y, max_depth = None):
    """
    Trains a decision tree on the input data using the information gain criterion
    (set the criterion in the constructor to 'entropy').
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
        max_depth (int): The maximum depth the decision tree is allowed to be. If
                         `None`, then the depth is unbounded.
    Returns:
        DecisionTreeClassifier: the learned decision tree.
    """
    clf = DecisionTreeClassifier(criterion= 'entropy', max_depth = max_depth)
    clf.fit(X, y)
    return clf


# In[648]:


from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def train_sgd(X, y):
    """
    Trains an `SGDClassifier` using 'log' loss on the input data.
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
        learning_rate (str): The learning rate to use. See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    Returns:
        SGDClassifier: the learned classifier.
    """
    
    clf = linear_model.SGDClassifier(loss = 'log')
    clf.fit(X,y)
    return clf


# In[649]:


import random
import math
def train_sgd_with_stumps(X, y):   
    
    """
    Trains an `SGDClassifier` using 'log' loss on the input data. The classifier will
    be trained on features that are computed using decision tree stumps.
    
    This function will return two items, the `SGDClassifier` and list of `DecisionTreeClassifier`s
    which were used to compute the new feature set. If `sgd` is the name of your `SGDClassifier`
    and `stumps` is the name of your list of `DecisionTreeClassifier`s, then writing
    `return sgd, stumps` will return both of them at the same time.
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
    Returns:
        SGDClassifier: the learned classifier.
        List[DecisionTree]: the decision stumps that were used to compute the features
                            for the `SGDClassifier`.
    """
    randomX = []
    randomY = []
    for i in random.sample(range(0, len(X)), math.floor(len(X)/2)):
        randomX.append(X[i])
        randomY.append(y[i])

    listOfTrees = []
    for i in range(0,200):       
        clf = DecisionTreeClassifier(criterion= 'entropy', max_depth = 8)
        clf.fit(randomX, randomY)
    
        listOfTrees.append(clf)
        
    
    newMatrix = np.zeros((len(X),200))
    for i in range(0, 200):
        pred = listOfTrees[i].predict(X)           
        for j in range(0, len(X)):           
            newMatrix[j][i] = pred[j]                     
       

    clf = train_sgd(newMatrix, y)
    
    return clf, listOfTrees


    
    
    
    
    


# In[650]:


# The input to this function can be an `SGDClassifier` or a `DecisionTreeClassifier`.
# Because they both use the same interface for predicting labels, the code can be the same
# for both of them.
def predict_clf(clf, X):
    """
    Predicts labels for all instances in `X` using the `clf` classifier. This function
    will be the same for `DecisionTreeClassifier`s and `SGDClassifier`s.
    
    Args:
        clf: (`SGDClassifier` or `DecisionTreeClassifier`): the trained classifier.
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
    Returns:
        List[int]: the predicted labels for each instance in `X`.
    """
    return clf.predict(X)  


# In[651]:


# The SGD-DT classifier can't use the same function as the SGD or decision trees
# because it requires an extra argument

def predict_sgd_with_stumps(sgd, stumps, X):
    """
    Predicts labels for all instances `X` using the `SGDClassifier` trained with
    features computed from decision stumps. The input `X` will be a matrix of the
    original features. The stumps will be used to map `X` from the original features
    to the features that the `SGDClassifier` were trained with.
    
    Args:
        sgd (`SGDClassifier`): the classifier that was trained with features computed
                               using the input stumps.
        stumps (List[DecisionTreeClassifier]): a list of `DecisionTreeClassifier`s that
                                               were used to train the `SGDClassifier`.
        X (list of lists): The features that were used to train the stumps (i.e. the original
                           feature set).
    Returns:
        List[int]: the predicted labels for each instance in `X`.
    """
    newMatrix = np.zeros((len(X),200))
    for i in range(0, 200):
        pred = stumps[i].predict(X)
        for j in range(0, len(X)):
            newMatrix[j][i] = pred[j]
 
    
    return sgd.predict(newMatrix)


# In[642]:



# print("Optimal sgd with eta = 100 \n")
# #array to hold the average of the 20 folds
# totalFoldAvg = []
# #array to hold the average of the fifth folds
# fiveFoldAvg = []

# #for-loop for the five folds
# for i in range(0,5): 
#     #X will hold the result of compute_features for the four folds
#     X = []
#     #y will hold the true label results for the four folds
#     y = []

#     #on each iteration of the for loop, change the folds
#     if i == 0:
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-3.txt', 'r')
     
#         fifth=   open('train.fold-4.txt', 'r')
 
#     if(i==1):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-3.txt', 'r')
     
#     if(i==2):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-2.txt', 'r')
     
#     if(i==3):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-1.txt', 'r')
     
#     if(i==4):
#         first =  open('train.fold-1.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-0.txt', 'r')
     
 
#     #for each of the following five blocks, we 
#     #go through each line in the four selected folds 
#     #and add the real label to y and the compute
#     #feature to X, we also add to our local fold
#     #array (for later testing on the folds)

#     firstArrayX = []
#     firstArrayY = []
#     for line in first:
#         if(line[0] == '+'):
#             y.append(1)
#             firstArrayY.append(1)
#         else:
#             y.append(0)
#             firstArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         firstArrayX.append(compute_features(line[1:len(line)]))
     
#     secondArrayX = []
#     secondArrayY = []
#     for line in second:
#         if(line[0] == '+'):
#             y.append(1)
#             secondArrayY.append(1)
#         else:
#             y.append(0)
#             secondArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         secondArrayX.append(compute_features(line[1:len(line)]))
     
#     thirdArrayX = []
#     thirdArrayY = []
#     for line in third:
#         if(line[0] == '+'):
#             y.append(1)
#             thirdArrayY.append(1)
#         else:
#             y.append(0)
#             thirdArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         thirdArrayX.append(compute_features(line[1:len(line)]))
     
#     fourthArrayX = []
#     fourthArrayY = []
#     for line in fourth:
#         if(line[0] == '+'):
#             y.append(1)
#             fourthArrayY.append(1)
#         else:
#             y.append(0)
#             fourthArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         fourthArrayX.append(compute_features(line[1:len(line)]))
     
#     fifthArrayX = []
#     fifthArrayY = []
#     for line in fifth:
#         if(line[0] == '+'):
#             fifthArrayY.append(1)
#         else:
#             fifthArrayY.append(0)
#         fifthArrayX.append(compute_features(line[1:len(line)]))
     
#     #train and fit the classifier
#     clf = train_sgd(X,y)
 
 
#      #predict the labels for the unlabelled data
#     label =  list(open('test.unlabeled.txt', 'r'))        
#     f = open("sgd.txt", "w")
#     newLab = []
#     for line in label:
#         newLab.append(compute_features(line))
#     for i in clf.predict(newLab):
#         if(i == 1):
#             f.write("+\n")
#         else:
#             f.write("-\n")
 
#     totalFoldAvg.append(clf.score(firstArrayX,firstArrayY))
#     totalFoldAvg.append(clf.score(secondArrayX,secondArrayY))
#     totalFoldAvg.append(clf.score(thirdArrayX,thirdArrayY))
#     totalFoldAvg.append(clf.score(fourthArrayX,fourthArrayY))
  


 
#     fiveFoldAvg.append(clf.score(fifthArrayX,fifthArrayY))

# ci = 4.604 * (np.std(fiveFoldAvg) / math.sqrt(5))
# print("Confidence interval for fifth fold: " + str(sum(fiveFoldAvg)/float(len(fiveFoldAvg))) + "+-" + str(ci))
# ci2 = 4.604 * (np.std(totalFoldAvg) / math.sqrt(20))
# print("Confidence interval for total fold: " + str(np.mean(totalFoldAvg)) + "+-" + str(ci2))



# print('')
# print('')
# print('')
# print('')
# print("Optimal sgd with eta = 1 \n")
# #array to hold the average of the 20 folds
# totalFoldAvg = []
# #array to hold the average of the fifth folds
# fiveFoldAvg = []

# #for-loop for the five folds
# for i in range(0,5): 
#     #X will hold the result of compute_features for the four folds
#     X = []
#     #y will hold the true label results for the four folds
#     y = []
 
 
#     #on each iteration of the for loop, change the folds
#     if i == 0:
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-3.txt', 'r')
     
#         fifth=   open('train.fold-4.txt', 'r')
 
#     if(i==1):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-3.txt', 'r')
     
#     if(i==2):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-2.txt', 'r')
     
#     if(i==3):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-1.txt', 'r')
     
#     if(i==4):
#         first =  open('train.fold-1.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-0.txt', 'r')
     
#     #for each of the following five blocks, we 
#     #go through each line in the four selected folds 
#     #and add the real label to y and the compute
#     #feature to X, we also add to our local fold
#     #array (for later testing on the folds)
 
#     firstArrayX = []
#     firstArrayY = []
#     for line in first:
#         if(line[0] == '+'):
#             y.append(1)
#             firstArrayY.append(1)
#         else:
#             y.append(0)
#             firstArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         firstArrayX.append(compute_features(line[1:len(line)]))
     
#     secondArrayX = []
#     secondArrayY = []
#     for line in second:
#         if(line[0] == '+'):
#             y.append(1)
#             secondArrayY.append(1)
#         else:
#             y.append(0)
#             secondArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         secondArrayX.append(compute_features(line[1:len(line)]))
     
#     thirdArrayX = []
#     thirdArrayY = []
#     for line in third:
#         if(line[0] == '+'):
#             y.append(1)
#             thirdArrayY.append(1)
#         else:
#             y.append(0)
#             thirdArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         thirdArrayX.append(compute_features(line[1:len(line)]))
     
#     fourthArrayX = []
#     fourthArrayY = []
#     for line in fourth:
#         if(line[0] == '+'):
#             y.append(1)
#             fourthArrayY.append(1)
#         else:
#             y.append(0)
#             fourthArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         fourthArrayX.append(compute_features(line[1:len(line)]))
     
#     fifthArrayX = []
#     fifthArrayY = []
#     for line in fifth:
#         if(line[0] == '+'):
#             fifthArrayY.append(1)
#         else:
#             fifthArrayY.append(0)
#         fifthArrayX.append(compute_features(line[1:len(line)]))

     
#     #train and fit the classifier
#     clf =linear_model.SGDClassifier(loss = 'log', learning_rate = 'optimal', eta0=1.0)
#     clf.fit(X,y)
 
#     totalFoldAvg.append(clf.score(firstArrayX,firstArrayY))
#     totalFoldAvg.append(clf.score(secondArrayX,secondArrayY))
#     totalFoldAvg.append(clf.score(thirdArrayX,thirdArrayY))
#     totalFoldAvg.append(clf.score(fourthArrayX,fourthArrayY))
  


 
#     fiveFoldAvg.append(clf.score(fifthArrayX,fifthArrayY))

# ci = 4.604 * (np.std(fiveFoldAvg) / math.sqrt(5))
# print("Confidence interval for fifth fold: " + str(sum(fiveFoldAvg)/float(len(fiveFoldAvg))) + "+-" + str(ci))
# ci2 = 4.604 * (np.std(totalFoldAvg) / math.sqrt(20))
# print("Confidence interval for total fold: " + str(np.mean(totalFoldAvg)) + "+-" + str(ci2))


# print('')
# print('')
# print('')
# print('')
# print("Constant sgd with eta = 100 \n")
# #array to hold the average of the 20 folds
# totalFoldAvg = []
# #array to hold the average of the fifth folds
# fiveFoldAvg = []

# #for-loop for the five folds
# for i in range(0,5): 
#     #X will hold the result of compute_features for the four folds
#     X = []
#     #y will hold the true label results for the four folds
#     y = []

#     #on each iteration of the for loop, change the folds
#     if i == 0:
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-3.txt', 'r')
     
#         fifth=   open('train.fold-4.txt', 'r')
 
#     if(i==1):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-3.txt', 'r')
     
#     if(i==2):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-2.txt', 'r')
     
#     if(i==3):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-1.txt', 'r')
     
#     if(i==4):
#         first =  open('train.fold-1.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-0.txt', 'r')
     
 
#     #for each of the following five blocks, we 
#     #go through each line in the four selected folds 
#     #and add the real label to y and the compute
#     #feature to X, we also add to our local fold
#     #array (for later testing on the folds)
#     firstArrayX = []
#     firstArrayY = []
#     for line in first:
#         if(line[0] == '+'):
#             y.append(1)
#             firstArrayY.append(1)
#         else:
#             y.append(0)
#             firstArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         firstArrayX.append(compute_features(line[1:len(line)]))
     
#     secondArrayX = []
#     secondArrayY = []
#     for line in second:
#         if(line[0] == '+'):
#             y.append(1)
#             secondArrayY.append(1)
#         else:
#             y.append(0)
#             secondArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         secondArrayX.append(compute_features(line[1:len(line)]))
     
#     thirdArrayX = []
#     thirdArrayY = []
#     for line in third:
#         if(line[0] == '+'):
#             y.append(1)
#             thirdArrayY.append(1)
#         else:
#             y.append(0)
#             thirdArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         thirdArrayX.append(compute_features(line[1:len(line)]))
     
#     fourthArrayX = []
#     fourthArrayY = []
#     for line in fourth:
#         if(line[0] == '+'):
#             y.append(1)
#             fourthArrayY.append(1)
#         else:
#             y.append(0)
#             fourthArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         fourthArrayX.append(compute_features(line[1:len(line)]))
     
#     fifthArrayX = []
#     fifthArrayY = []
#     for line in fifth:
#         if(line[0] == '+'):
#             fifthArrayY.append(1)
#         else:
#             fifthArrayY.append(0)
#         fifthArrayX.append(compute_features(line[1:len(line)]))

#     #train and fit the classifier
#     clf = linear_model.SGDClassifier(loss = 'log', learning_rate = 'constant', eta0 = 100.0)
#     clf.fit(X,y)
 
#     totalFoldAvg.append(clf.score(firstArrayX,firstArrayY))
#     totalFoldAvg.append(clf.score(secondArrayX,secondArrayY))
#     totalFoldAvg.append(clf.score(thirdArrayX,thirdArrayY))
#     totalFoldAvg.append(clf.score(fourthArrayX,fourthArrayY))
  


 
#     fiveFoldAvg.append(clf.score(fifthArrayX,fifthArrayY))

# ci = 4.604 * (np.std(fiveFoldAvg) / math.sqrt(5))
# print("Confidence interval for fifth fold: " + str(sum(fiveFoldAvg)/float(len(fiveFoldAvg))) + "+-" + str(ci))
# ci2 = 4.604 * (np.std(totalFoldAvg) / math.sqrt(20))
# print("Confidence interval for total fold: " + str(np.mean(totalFoldAvg)) + "+-" + str(ci2))
 
 
 
# print('')
# print('')
# print('')
# print('')
# print("Four tree \n")
# #array to hold the average of the 20 folds
# totalFoldAvg = []
# #array to hold the average of the fifth folds
# fiveFoldAvg = []

# #for-loop for the five folds
# for i in range(0,5): 
#     #X will hold the result of compute_features for the four folds
#     X = []
#     #y will hold the true label results for the four folds
#     y = []
 
#      #on each iteration of the for loop, change the folds
#     if i == 0:
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-3.txt', 'r')
     
#         fifth=   open('train.fold-4.txt', 'r')
 
#     if(i==1):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-3.txt', 'r')
     
#     if(i==2):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-2.txt', 'r')
     
#     if(i==3):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-1.txt', 'r')
     
#     if(i==4):
#         first =  open('train.fold-1.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-0.txt', 'r')

     
#     #for each of the following five blocks, we 
#     #go through each line in the four selected folds 
#     #and add the real label to y and the compute
#     #feature to X, we also add to our local fold
#     #array (for later testing on the folds)
 
#     firstArrayX = []
#     firstArrayY = []
#     for line in first:
#         if(line[0] == '+'):
#             y.append(1)
#             firstArrayY.append(1)
#         else:
#             y.append(0)
#             firstArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         firstArrayX.append(compute_features(line[1:len(line)]))
     
#     secondArrayX = []
#     secondArrayY = []
#     for line in second:
#         if(line[0] == '+'):
#             y.append(1)
#             secondArrayY.append(1)
#         else:
#             y.append(0)
#             secondArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         secondArrayX.append(compute_features(line[1:len(line)]))
     
#     thirdArrayX = []
#     thirdArrayY = []
#     for line in third:
#         if(line[0] == '+'):
#             y.append(1)
#             thirdArrayY.append(1)
#         else:
#             y.append(0)
#             thirdArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         thirdArrayX.append(compute_features(line[1:len(line)]))
     
#     fourthArrayX = []
#     fourthArrayY = []
#     for line in fourth:
#         if(line[0] == '+'):
#             y.append(1)
#             fourthArrayY.append(1)
#         else:
#             y.append(0)
#             fourthArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         fourthArrayX.append(compute_features(line[1:len(line)]))
     
#     fifthArrayX = []
#     fifthArrayY = []
#     for line in fifth:
#         if(line[0] == '+'):
#             fifthArrayY.append(1)
#         else:
#             fifthArrayY.append(0)
#         fifthArrayX.append(compute_features(line[1:len(line)]))
     
#      #train and fit the classifier
#     clf = DecisionTreeClassifier(criterion= 'entropy', max_depth = 4)
#     clf.fit(X,y)
 
 
 
#      #predict the labels for the unlabelled data
#     label =  list(open('test.unlabeled.txt', 'r'))        
#     f = open("dt-4.txt", "w")
#     newLab = []
#     for line in label:
#         newLab.append(compute_features(line))
#     for i in clf.predict(newLab):
#         if(i == 1):
#             f.write("+\n")
#         else:
#             f.write("-\n")


#     totalFoldAvg.append(clf.score(firstArrayX,firstArrayY))
#     totalFoldAvg.append(clf.score(secondArrayX,secondArrayY))
#     totalFoldAvg.append(clf.score(thirdArrayX,thirdArrayY))
#     totalFoldAvg.append(clf.score(fourthArrayX,fourthArrayY))



 
#     fiveFoldAvg.append(clf.score(fifthArrayX,fifthArrayY))
 

# ci = 4.604 * (np.std(fiveFoldAvg) / math.sqrt(5))
# print("Confidence interval for fifth fold: " + str(sum(fiveFoldAvg)/float(len(fiveFoldAvg))) + "+-" + str(ci))
# ci2 = 4.604 * (np.std(totalFoldAvg) / math.sqrt(20))
# print("Confidence interval for total fold: " + str(np.mean(totalFoldAvg)) + "+-" + str(ci2))  

# print('')
# print('')
# print('')
# print('')
# print("Eight tree \n")
# #array to hold the average of the 20 folds
# totalFoldAvg = []
# #array to hold the average of the fifth folds
# fiveFoldAvg = []

# #for-loop for the five folds
# for i in range(0,5): 
#     #X will hold the result of compute_features for the four folds
#     X = []
#     #y will hold the true label results for the four folds
#     y = []
 
#      #on each iteration of the for loop, change the folds
#     if i == 0:
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-3.txt', 'r')
     
#         fifth=   open('train.fold-4.txt', 'r')
 
#     if(i==1):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-3.txt', 'r')
     
#     if(i==2):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-2.txt', 'r')
     
#     if(i==3):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-1.txt', 'r')
     
#     if(i==4):
#         first =  open('train.fold-1.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-0.txt', 'r')

#     #for each of the following five blocks, we 
#     #go through each line in the four selected folds 
#     #and add the real label to y and the compute
#     #feature to X, we also add to our local fold
#     #array (for later testing on the folds)
     
#     firstArrayX = []
#     firstArrayY = []
#     for line in first:
#         if(line[0] == '+'):
#             y.append(1)
#             firstArrayY.append(1)
#         else:
#             y.append(0)
#             firstArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         firstArrayX.append(compute_features(line[1:len(line)]))
     
#     secondArrayX = []
#     secondArrayY = []
#     for line in second:
#         if(line[0] == '+'):
#             y.append(1)
#             secondArrayY.append(1)
#         else:
#             y.append(0)
#             secondArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         secondArrayX.append(compute_features(line[1:len(line)]))
     
#     thirdArrayX = []
#     thirdArrayY = []
#     for line in third:
#         if(line[0] == '+'):
#             y.append(1)
#             thirdArrayY.append(1)
#         else:
#             y.append(0)
#             thirdArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         thirdArrayX.append(compute_features(line[1:len(line)]))
     
#     fourthArrayX = []
#     fourthArrayY = []
#     for line in fourth:
#         if(line[0] == '+'):
#             y.append(1)
#             fourthArrayY.append(1)
#         else:
#             y.append(0)
#             fourthArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         fourthArrayX.append(compute_features(line[1:len(line)]))
     
#     fifthArrayX = []
#     fifthArrayY = []
#     for line in fifth:
#         if(line[0] == '+'):
#             fifthArrayY.append(1)
#         else:
#             fifthArrayY.append(0)
#         fifthArrayX.append(compute_features(line[1:len(line)]))
     
#     #train and fit the classifier
#     clf = DecisionTreeClassifier(criterion= 'entropy', max_depth = 8)
#     clf.fit(X, y)
 
 
#      #predict the labels for the unlabelled data
#     label =  list(open('test.unlabeled.txt', 'r'))        
#     f = open("dt-8.txt", "w")
#     newLab = []
#     for line in label:
#         newLab.append(compute_features(line))
#     for i in clf.predict(newLab):
#         if(i == 1):
#             f.write("+\n")
#         else:
#             f.write("-\n")
 


#     totalFoldAvg.append(clf.score(firstArrayX,firstArrayY))
#     totalFoldAvg.append(clf.score(secondArrayX,secondArrayY))
#     totalFoldAvg.append(clf.score(thirdArrayX,thirdArrayY))
#     totalFoldAvg.append(clf.score(fourthArrayX,fourthArrayY))
 
#     fiveFoldAvg.append(clf.score(fifthArrayX,fifthArrayY))


# ci = 4.604 * (np.std(fiveFoldAvg) / math.sqrt(5))
# print("Confidence interval for fifth fold: " + str(sum(fiveFoldAvg)/float(len(fiveFoldAvg))) + "+-" + str(ci))
# ci2 = 4.604 * (np.std(totalFoldAvg) / math.sqrt(20))
# print("Confidence interval for total fold: " + str(np.mean(totalFoldAvg)) + "+-" + str(ci2))


# print('')
# print('')
# print('')
# print('')
# print("Full tree \n")
# #array to hold the average of the 20 folds
# totalFoldAvg = []
# #array to hold the average of the fifth folds
# fiveFoldAvg = []

# #for-loop for the five folds
# for i in range(0,5): 
#     #X will hold the result of compute_features for the four folds
#     X = []
#     #y will hold the true label results for the four folds
#     y = []

#      #on each iteration of the for loop, change the folds
#     if i == 0:
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-3.txt', 'r')
     
#         fifth=   open('train.fold-4.txt', 'r')
 
#     if(i==1):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-2.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-3.txt', 'r')
     
#     if(i==2):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-1.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-2.txt', 'r')
     
#     if(i==3):
#         first =  open('train.fold-0.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-1.txt', 'r')
     
#     if(i==4):
#         first =  open('train.fold-1.txt', 'r')
#         second = open('train.fold-2.txt', 'r')
#         third =  open('train.fold-3.txt', 'r')
#         fourth =  open('train.fold-4.txt', 'r')
     
#         fifth=   open('train.fold-0.txt', 'r')

     
#     #for each of the following five blocks, we 
#     #go through each line in the four selected folds 
#     #and add the real label to y and the compute
#     #feature to X, we also add to our local fold
#     #array (for later testing on the folds)
#     firstArrayX = []
#     firstArrayY = []
#     for line in first:
#         if(line[0] == '+'):
#             y.append(1)
#             firstArrayY.append(1)
#         else:
#             y.append(0)
#             firstArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         firstArrayX.append(compute_features(line[1:len(line)]))
     
#     secondArrayX = []
#     secondArrayY = []
#     for line in second:
#         if(line[0] == '+'):
#             y.append(1)
#             secondArrayY.append(1)
#         else:
#             y.append(0)
#             secondArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         secondArrayX.append(compute_features(line[1:len(line)]))
     
#     thirdArrayX = []
#     thirdArrayY = []
#     for line in third:
#         if(line[0] == '+'):
#             y.append(1)
#             thirdArrayY.append(1)
#         else:
#             y.append(0)
#             thirdArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         thirdArrayX.append(compute_features(line[1:len(line)]))
     
#     fourthArrayX = []
#     fourthArrayY = []
#     for line in fourth:
#         if(line[0] == '+'):
#             y.append(1)
#             fourthArrayY.append(1)
#         else:
#             y.append(0)
#             fourthArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         fourthArrayX.append(compute_features(line[1:len(line)]))
     
#     fifthArrayX = []
#     fifthArrayY = []
#     for line in fifth:
#         if(line[0] == '+'):
#             fifthArrayY.append(1)
#         else:
#             fifthArrayY.append(0)
#         fifthArrayX.append(compute_features(line[1:len(line)]))
 
#     #train and fit the classifier
#     clf =  DecisionTreeClassifier(criterion= 'entropy', max_depth = 100)
#     clf.fit(X, y)
 
 
#      #predict the labels for the unlabelled data
#     label =  list(open('test.unlabeled.txt', 'r'))
#     f = open("dt.txt", "w")
#     newLab = []
#     for line in label:
#         newLab.append(compute_features(line))
#     for i in clf.predict(newLab):
#         if(i == 1):
#             f.write("+\n")
#         else:
#             f.write("-\n")
 
 
   


#     totalFoldAvg.append(clf.score(firstArrayX,firstArrayY))
#     totalFoldAvg.append(clf.score(secondArrayX,secondArrayY))
#     totalFoldAvg.append(clf.score(thirdArrayX,thirdArrayY))
#     totalFoldAvg.append(clf.score(fourthArrayX,fourthArrayY))
 
#     fiveFoldAvg.append(clf.score(fifthArrayX,fifthArrayY))


# ci = 4.604 * (np.std(fiveFoldAvg) / math.sqrt(5))
# print("Confidence interval for fifth fold: " + str(sum(fiveFoldAvg)/float(len(fiveFoldAvg))) + "+-" + str(ci))
# ci2 = 4.604 * (np.std(totalFoldAvg) / math.sqrt(20))
# print("Confidence interval for total fold: " + str(np.mean(totalFoldAvg)) + "+-" + str(ci2))



# print('')
# print('')
# print('')
# print('')

# print("200 tree SGD \n")
# #array to hold the average of the 20 folds
# totalFoldAvg = []
# #array to hold the average of the fifth folds
# fiveFoldAvg = []

# #for-loop for the five folds
# for i in range(0,5): 
#     #X will hold the result of compute_features for the four folds
#     X = []
#     #y will hold the true label results for the four folds
#     y = []
 
#      #on each iteration of the for loop, change the folds
#     if i == 0:
#         first =  list(open('train.fold-0.txt', 'r'))
#         second = list(open('train.fold-1.txt', 'r'))
#         third =  list(open('train.fold-2.txt', 'r'))
#         fourth =  list(open('train.fold-3.txt', 'r'))
     
#         fifth=   list(open('train.fold-4.txt', 'r'))
 
#     if(i==1):
#         first =  list(open('train.fold-0.txt', 'r'))
#         second = list(open('train.fold-1.txt', 'r'))
#         third =  list(open('train.fold-2.txt', 'r'))
#         fourth =  list(open('train.fold-4.txt', 'r'))
     
#         fifth=   list(open('train.fold-3.txt', 'r'))
     
#     if(i==2):
#         first =  list(open('train.fold-0.txt', 'r'))
#         second = list(open('train.fold-1.txt', 'r'))
#         third =  list(open('train.fold-3.txt', 'r'))
#         fourth =  list(open('train.fold-4.txt', 'r'))
     
#         fifth=   list(open('train.fold-2.txt', 'r'))
     
#     if(i==3):
#         first =  list(open('train.fold-0.txt', 'r'))
#         second = list(open('train.fold-2.txt', 'r'))
#         third =  list(open('train.fold-3.txt', 'r'))
#         fourth =  list(open('train.fold-4.txt', 'r'))
     
#         fifth=   list(open('train.fold-1.txt', 'r'))
     
#     if(i==4):
#         first =  list(open('train.fold-1.txt', 'r'))
#         second = list(open('train.fold-2.txt', 'r'))
#         third =  list(open('train.fold-3.txt', 'r'))
#         fourth =  list(open('train.fold-4.txt', 'r'))
     
#         fifth=   list(open('train.fold-0.txt', 'r'))
     
    
#     #for each of the following five blocks, we 
#     #go through each line in the four selected folds 
#     #and add the real label to y and the compute
#     #feature to X, we also add to our local fold
#     #array (for later testing on the folds)
#     firstArrayX = []
#     firstArrayY = []
#     for line in first:
#         if(line[0] == '+'):
#             y.append(1)
#             firstArrayY.append(1)
#         else:
#             y.append(0)
#             firstArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         firstArrayX.append(compute_features(line[1:len(line)]))
     
#     secondArrayX = []
#     secondArrayY = []
#     for line in second:
#         if(line[0] == '+'):
#             y.append(1)
#             secondArrayY.append(1)
#         else:
#             y.append(0)
#             secondArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         secondArrayX.append(compute_features(line[1:len(line)]))
     
#     thirdArrayX = []
#     thirdArrayY = []
#     for line in third:
#         if(line[0] == '+'):
#             y.append(1)
#             thirdArrayY.append(1)
#         else:
#             y.append(0)
#             thirdArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         thirdArrayX.append(compute_features(line[1:len(line)]))
     
#     fourthArrayX = []
#     fourthArrayY = []
#     for line in fourth:
#         if(line[0] == '+'):
#             y.append(1)
#             fourthArrayY.append(1)
#         else:
#             y.append(0)
#             fourthArrayY.append(0)
#         X.append(compute_features(line[1:len(line)]))
#         fourthArrayX.append(compute_features(line[1:len(line)]))
     
#     fifthArrayX = []
#     fifthArrayY = []
#     for line in fifth:
#         if(line[0] == '+'):
#             fifthArrayY.append(1)
#         else:
#             fifthArrayY.append(0)
#         fifthArrayX.append(compute_features(line[1:len(line)]))
     
#     #train and fit the classifier   
#     clf, treeList = train_sgd_with_stumps(X, y) 
 
 
#     #predict the labels for the unlabelled data
#     label =  list(open('test.unlabeled.txt', 'r'))        
#     f = open("sgd-dt.txt", "w")
#     newLab = []
#     for line in label:
#         newLab.append(compute_features(line))
#     for i in predict_sgd_with_stumps(clf, treeList, newLab):
#         if(i == 1):
#             f.write("+\n")
#         else:
#             f.write("-\n")
 
#     #manually compute the accuracy for each of the five folds individually
#     right = 0
#     sum1 = 0    
#     tempList = predict_sgd_with_stumps(clf, treeList, firstArrayX)

 

 #    for i in range(0, len(tempList)):
#         if(firstArrayY[i] == tempList[i]):
#             right = right + 1
#         sum1 = sum1 + 1

 
#     #add the average to the array containing all the 20 folds to average later
#     totalFoldAvg.append(right/sum1)
 
 
#     right = 0
#     sum1 = 0    
#     tempList = predict_sgd_with_stumps(clf, treeList, secondArrayX)
 
#     for i in range(0, len(tempList)):
#         if(secondArrayY[i] == tempList[i]):
#             right = right + 1
#         sum1 = sum1 + 1
#     totalFoldAvg.append(right/sum1)

 
 
#     right = 0
#     sum1 = 0    
#     tempList = predict_sgd_with_stumps(clf, treeList, thirdArrayX)
 
#     for i in range(0, len(tempList)):
#         if(thirdArrayY[i] == tempList[i]):
#             right = right + 1
#         sum1 = sum1 + 1
#     totalFoldAvg.append(right/sum1)

 
 
 
#     right = 0
#     sum1 = 0    
#     tempList = predict_sgd_with_stumps(clf, treeList, fourthArrayX)
 
#     for i in range(0, len(tempList)):
#         if(fourthArrayY[i] == tempList[i]):
#             right = right + 1
#         sum1 = sum1 + 1

#     totalFoldAvg.append(right/sum1)

 

 
#     #compute the fifth, unseen fold accuracy
#     right = 0
#     sum1 = 0    
#     tempList = predict_sgd_with_stumps(clf, treeList, fifthArrayX)
 
#     for i in range(0, len(tempList)):
#         if(fifthArrayY[i] == tempList[i]):
#             right = right + 1
#         sum1 = sum1 + 1
 
#     fiveFoldAvg.append(right/ sum1)
 

# ci = 4.604 * (np.std(fiveFoldAvg) / math.sqrt(5))
# print("Confidence interval for fifth: " + str(sum(fiveFoldAvg)/float(len(fiveFoldAvg))) + "+-" + str(ci))
# ci2 = 4.604 * (np.std(totalFoldAvg) / math.sqrt(20))
# print("Confidence interval for total fold: " + str(np.mean(totalFoldAvg)) + "+-" + str(ci2))

 

