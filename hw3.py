
# coding: utf-8

# # Homework 3 Template
# This is the template for the third homework assignment.
# Below are some class and function templates which we require you to fill out.
# These will be tested by the autograder, so it is important to not edit the function definitions.
# The functions have python docstrings which should indicate what the input and output arguments are.

# ## Instructions for the Autograder
# When you submit your code to the autograder on Gradescope, you will need to comment out any code which is not an import statement or contained within a function definition.

# In[2]:


############################################################
# Imports
############################################################
# Include your imports here, if any are used.
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms


# In[3]:


def extract_data(x_data_filepath, y_data_filepath):
    X = np.load(x_data_filepath)
    y = np.load(y_data_filepath)
    return X, y


# In[4]:


############################################################
# Extracting and loading data
############################################################
class Dataset(Dataset):
    """CIFAR-10 image dataset."""
    def __init__(self, X, y, transformations=None):
        self.len = len(X)           
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# In[5]:


############################################################
# Feed Forward Neural Network
############################################################
class FeedForwardNN(nn.Module):
    """ 
        (1) Use self.fc1 as the variable name for your first fully connected layer
        (2) Use self.fc2 as the variable name for your second fully connected layer
    """
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 1500)
        self.fc2 = nn.Linear(1500, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)        
        out = F.sigmoid(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

    """ 
        Please do not change the functions below. 
        They will be used to test the correctness of your implementation 
    """
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()


# In[6]:


############################################################
# Convolutional Neural Network
############################################################
class ConvolutionalNN(nn.Module):
    """ 
        (1) Use self.conv1 as the variable name for your first convolutional layer
        (2) Use self.pool as the variable name for your pooling layer
        (3) User self.conv2 as the variable name for your second convolutional layer
        (4) Use self.fc1 as the variable name for your first fully connected layer
        (5) Use self.fc2 as the variable name for your second fully connected layer
        (6) Use self.fc3 as the variable name for your third fully connected layer
    """
    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 7, 3, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(7, 16, 3, 1, 0)
        self.fc1 = nn.Linear(16*13*13, 130)
        self.fc2 = nn.Linear(130, 72)
        self.fc3 = nn.Linear(72, 10)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        
        return out
    
    """ 
        Please do not change the functions below. 
        They will be used to test the correctness of your implementation 
    """
    def get_conv1_params(self):
        return self.conv1.__repr__()
    
    def get_pool_params(self):
        return self.pool.__repr__()

    def get_conv2_params(self):
        return self.conv2.__repr__()
    
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()
    
    def get_fc3_params(self):
        return self.fc3.__repr__()


# In[7]:


############################################################
# Hyperparameterized Feed Forward Neural Network
############################################################
class HyperParamsFeedForwardNN(nn.Module):
     def __init__(self):
        super(HyperParamsFeedForwardNN, self).__init__()
        #More output channels
        self.fc1 = nn.Linear(32*32*3, 3000)
        #More input chanells
        self.fc2 = nn.Linear(3000, 10)
        
     def forward(self, x):
        out = x.view(x.size(0), -1)        
        out = F.sigmoid(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


# In[8]:


############################################################
# Hyperparameterized Convolutional Neural Network
############################################################
class HyperParamsConvNN(nn.Module):
    def __init__(self, kernel_size=3, img_size=32):
        super(HyperParamsConvNN, self).__init__()
        #uniformly apply the given kernel_size to both the convolutional and poleld layers
        self.conv1 = nn.Conv2d(3, 7, kernel_size, 1, 0)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(7, 16, kernel_size, 1, 0)
        self.fc1 = nn.Linear(16*13*13, 1550)
        self.fc2 = nn.Linear(1550, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        
        return out
    


# In[9]:


############################################################
# Run Experiment
############################################################
def run_experiment(neural_network, train_loader, test_loader, loss_function, optimizer):
    """
    Runs experiment on the model neural network given a train and test data loader, loss function and optimizer.

    Args:
        neural_network (NN model that extends torch.nn.Module): For example, it should take an instance of either
                                                                FeedForwardNN or ConvolutionalNN,
        train_loader (DataLoader),
        test_loader (DataLoader),
        loss_function (torch.nn.CrossEntropyLoss),
        optimizer (optim.SGD)
    Returns:
        tuple: First position, testing accuracy.
               Second position, training accuracy.
               Third position, training loss.

               For example, if you find that
                            testing accuracy = 0.76,
                            training accuracy = 0.24
                            training loss = 0.56

               This function should return (0.76, 0.24, 0.56)
    """
    max_epochs = 100

    loss_np = np.zeros((max_epochs))
    accuracy = np.zeros((max_epochs))
    test_accuracy = np.zeros((max_epochs))


    for epoch in range(max_epochs):
        temp_accuracy_train = np.zeros(train_loader.__len__())
        temp_loss_np = np.zeros(train_loader.__len__())
        print(epoch+1)
        for i, data in enumerate(train_loader, 0):

            # Get inputs and labels from data loader 
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # Feed the input data into the network        
            y_pred = neural_network(inputs)
            loss = loss_function(y_pred, labels)

            # zero gradient
            optimizer.zero_grad()

            # backpropogates to compute gradient
            loss.backward()

            # updates the weghts
            optimizer.step()

            # convert predicted laels into numpy
            y_pred_np = y_pred.data.numpy()
            label_np = labels.data.numpy()  
            
            correct = 0  
            for j in range(0,len(y_pred_np)):
                if np.argmax(y_pred_np[j]) == (label_np[j]):
                    correct += 1         
            
            temp_accuracy_train[i] = float(correct)/float(len(label_np))
            temp_loss_np[i] = loss.data.numpy()


        accuracy[epoch] = np.mean(temp_accuracy_train)
        loss_np[epoch] = np.mean(temp_loss_np)
            
        temp_accuracy_test = np.zeros(test_loader.__len__())
        for i, data in enumerate(test_loader, 0):

            # Get inputs and labels from data loader 
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # Feed the input data into the network 
            y_pred = neural_network(inputs)

            # convert predicted laels into numpy
            y_pred_np = y_pred.data.numpy()
            label_np = labels.data.numpy()

            correct = 0

            for j in range(0,len(y_pred_np)):
                if np.argmax(y_pred_np[j]) == (label_np[j]):
                    correct += 1


            temp_accuracy_test[i] = float(correct)/float(len(label_np))
            
        test_accuracy[epoch] = np.mean(temp_accuracy_test)
            
    print("final training accuracy: ", accuracy[max_epochs-1])
    print("final test accuracy: ", test_accuracy[max_epochs-1])


    epoch_number = np.arange(0,max_epochs,1)

    # Plot the loss over epoch
    plt.figure()
    plt.plot(epoch_number, loss_np)
    plt.title('loss over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')

    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, accuracy)
    plt.title('training accuracy over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    
    
        # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, test_accuracy)
    plt.title('test accuracy over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    
    return np.mean(test_accuracy), np.mean(accuracy), np.mean(loss_np)
#for test don't need the zer_grad or backward or step


# In[10]:


# #Experiment 1
# #CHANGE to average

# ex,ey = extract_data('train_images.npy', 'train_labels.npy')
# train_dataset = Dataset(ex, ey)
# ex,ey = extract_data('test_images.npy', 'test_labels.npy')
# test_dataset = Dataset(ex, ey)

# net =  FeedForwardNN()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.90)



# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# In[11]:


# #Experiment 2

# ex,ey = extract_data('train_images.npy', 'train_labels.npy')
# train_dataset = Dataset(ex, ey)
# ex,ey = extract_data('test_images.npy', 'test_labels.npy')
# test_dataset = Dataset(ex, ey)

# net =  ConvolutionalNN()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.90)



# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# In[12]:


#3 x 32 x 32, so each is row 
def normalize_image(image):
    """
    Normalizes the RGB pixel values of an image.

    Args:
        image (3D NumPy array): For example, it should take in a single 3x32x32 image from the CIFAR-10 dataset
    Returns:
        tuple: The normalized image        
    """
    redVal = image[0].flatten()
    greenVal = image[1].flatten()
    blueVal = image[2].flatten()  

    red_std = np.std(redVal)
    blue_std = np.std(blueVal)
    green_std = np.std(greenVal)

    red_mean = np.mean(redVal)
    blue_mean = np.mean(blueVal)
    green_mean = np.mean(greenVal)


    for i in range(0, len(image[0])):
        for j in range(0, len(image[0][0])):
            image[0][i][j] = (image[0][i][j] - red_mean) / red_std
    for i in range(0, len(image[1])):
        for j in range(0, len(image[1][0])):
            image[1][i][j] = (image[1][i][j] - green_mean) / green_std
    for i in range(0, len(image[2])):
        for j in range(0, len(image[2][0])):
            image[2][i][j] = (image[2][i][j] - blue_mean) / blue_std


    return image


# In[13]:


# #Experiment 3


# train_ex,train_ey = extract_data('train_images.npy', 'train_labels.npy')
# for i in range(0, len(train_ex)):
#     train_ex[i] = normalize_image(train_ex[i])
# train_dataset = Dataset(train_ex, train_ey)


# ex,ey = extract_data('test_images.npy', 'test_labels.npy')
# for i in range(0, len(ex)):
#     ex[i] = normalize_image(ex[i])
# test_dataset = Dataset(ex, ey)



# net = ConvolutionalNN()
# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.90)
# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# In[14]:


# #Experiment 4
# #For hyper change batch_size, and lr, and momentum and kernel size and number of neurons


# ex,ey = extract_data('train_images.npy', 'train_labels.npy')
# train_dataset = Dataset(ex, ey)
# ex,ey = extract_data('test_images.npy', 'test_labels.npy')
# test_dataset = Dataset(ex, ey)


# net =  HyperParamsFeedForwardNN()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.90)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)



# net =  HyperParamsConvNN()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.90)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# In[22]:


# ############################################################
# # Feed Forward Neural Network
# ############################################################
# class FeedForwardNNEC(nn.Module):
#     """ 
#         (1) Use self.fc1 as the variable name for your first fully connected layer
#         (2) Use self.fc2 as the variable name for your second fully connected layer
#     """
#     def __init__(self):
#         super(FeedForwardNNEC, self).__init__()
#         self.fc1 = nn.Linear(32*32*3, 1500)
#         self.fc2 = nn.Linear(1500, 10)

#     def forward(self, x):
#         out = x.view(x.size(0), -1)        
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         return out

#     """ 
#         Please do not change the functions below. 
#         They will be used to test the correctness of your implementation 
#     """
#     def get_fc1_params(self):
#         return self.fc1.__repr__()
    
#     def get_fc2_params(self):
#         return self.fc2.__repr__()
# #Experiment EC


# ex,ey = extract_data('train_images.npy', 'train_labels.npy')
# train_dataset = Dataset(ex, ey)
# ex,ey = extract_data('test_images.npy', 'test_labels.npy')
# test_dataset = Dataset(ex, ey)


# # net =  FeedForwardNN()

# # train_loader = DataLoader(dataset=train_dataset,
# #                           batch_size=64)
# # test_loader = DataLoader(dataset=test_dataset,
# #                           batch_size=64)
# # loss_function = nn.CrossEntropyLoss()
# # optimizer = optim.Adagrad(net.parameters(), lr=0.001)


# # run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# # net =  FeedForwardNN()

# # train_loader = DataLoader(dataset=train_dataset,
# #                           batch_size=64)
# # test_loader = DataLoader(dataset=test_dataset,
# #                           batch_size=64)
# # loss_function = nn.CrossEntropyLoss()
# # optimizer = optim.Adamax(net.parameters(), lr=0.001)


# # run_experiment(net, train_loader, test_loader, loss_function, optimizer)



# # net =  FeedForwardNN()

# # train_loader = DataLoader(dataset=train_dataset,
# #                           batch_size=64)
# # test_loader = DataLoader(dataset=test_dataset,
# #                           batch_size=64)
# # loss_function = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(net.parameters(), lr=0.001)


# # run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# # net =  FeedForwardNN()

# # train_loader = DataLoader(dataset=train_dataset,
# #                           batch_size=64)
# # test_loader = DataLoader(dataset=test_dataset,
# #                           batch_size=64)
# # loss_function = nn.CrossEntropyLoss()
# # optimizer = optim.ASGD(net.parameters(), lr=0.001, alpha=0.75)


# # run_experiment(net, train_loader, test_loader, loss_function, optimizer)





















# net =  FeedForwardNNEC()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adagrad(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# net =  FeedForwardNNEC()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adamax(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)



# net =  FeedForwardNNEC()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)

      
# net =  FeedForwardNNEC()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.ASGD(net.parameters(), lr=0.001, alpha=0.75)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# In[23]:


# class ConvolutionalNNEC(nn.Module):
#     """ 
#         (1) Use self.conv1 as the variable name for your first convolutional layer
#         (2) Use self.pool as the variable name for your pooling layer
#         (3) User self.conv2 as the variable name for your second convolutional layer
#         (4) Use self.fc1 as the variable name for your first fully connected layer
#         (5) Use self.fc2 as the variable name for your second fully connected layer
#         (6) Use self.fc3 as the variable name for your third fully connected layer
#     """
#     def __init__(self):
#         super(ConvolutionalNNEC, self).__init__()
#         self.conv1 = nn.Conv2d(3, 7, 3, 1, 0)
#         self.pool = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(7, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16*13*13, 130)
#         self.fc2 = nn.Linear(130, 72)
#         self.fc3 = nn.Linear(72, 10)
#     def forward(self, x):
#         out = F.sigmoid(self.conv1(x))
#         out = self.pool(out)
#         out = F.sigmoid(self.conv2(out))
        
#         out = out.view(out.size(0), -1)
#         out = F.tanh(self.fc1(out))
#         out = F.tanh(self.fc2(out))
#         out = F.tanh(self.fc3(out))
        
#         return out
    
# #Experiment EC for CNN


# ex,ey = extract_data('train_images.npy', 'train_labels.npy')
# train_dataset = Dataset(ex, ey)
# ex,ey = extract_data('test_images.npy', 'test_labels.npy')
# test_dataset = Dataset(ex, ey)


# net =  ConvolutionalNN()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adagrad(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# net =  ConvolutionalNN()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adamax(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)



# net =  ConvolutionalNN()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# net =  ConvolutionalNN()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.ASGD(net.parameters(), lr=0.001, alpha=0.75)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)





















# net =  ConvolutionalNNEC()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adagrad(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# net =  ConvolutionalNNEC()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adamax(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)



# net =  ConvolutionalNNEC()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)


# net =  ConvolutionalNNEC()

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=64)
# test_loader = DataLoader(dataset=test_dataset,
#                           batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.ASGD(net.parameters(), lr=0.001, alpha=0.75)


# run_experiment(net, train_loader, test_loader, loss_function, optimizer)

