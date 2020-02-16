## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y)    pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        ###NAIMISHNET ARCHITECTURE: consists of 4convolution2D layers, 4MAXPOOLING2D layers, and 3 dense layers,         #with sandwitch dropout and activation layers.
        #Filter details of convolution layers:
        #layer name :::: #of filters ::: filter shape
        #conv2d_1   :::     32       ::: (4,4)
        #conv2d_2   :::     64       ::: (3,3)
        #conv2d_3   :::     128      ::: (2,2)
        #conv2d_4   :::     256      ::: (1,1)
        
        #Activation1 to Activation5 use ELU, Whereas Activation6 use RELU.
        #Dropout probability increases from 0.1 to 0.6 with step 0.1
        #Maxpooling2d_1 to Maxpooling2d_4 use a pool of shape(2,2) with no-overlapping strides and no zero               #padding
        #and rounded down.
        #convolution don't use zero padding, weights are initialized with uniform distribution.
        #dense layers weights are initialized using Glorot uniform initialization.
        #Adam optimizer is used for minimizing Mean Square Error (MSE).
        
        #The input is grayscaled image with shape  1*224*224
        
        #the outputs shapes are calculated as below:
        #For convolution layer:
        #w,h_out= w,h_in-m+2p/s + 1
        #where m is filter size, s is stride and p is padding.
        #For maxpooling layer:
        #w,h_out= w,h_in-n/s + 1
        #where n is the pool size and s is stride.
        
        #Convolution layers:
        #self.conv=nn.conv2d(input_depth,output_depth,kernel_size)
        #self.conv1=nn.Conv2d(1,32,4)     #32x221x221
        #self.conv2=nn.Conv2d(32,64,3)    #64x108x108
        #self.conv3=nn.Conv2d(64,128,2)    #128x53x53
        #self.conv4=nn.Conv2d(128,256,1)  #256x26x26
        
        #Maxpool layers:
        #self.pool=nn.MaxPool2d(kernel_size,stride)
        #self.pool1 = nn.MaxPool2d(2, 2)  #32x110x110
        #self.pool2 = nn.MaxPool2d(2, 2)  #64x54x54
        #self.pool3 = nn.MaxPool2d(2, 2)  #128x26x26
        #self.pool4 = nn.MaxPool2d(2, 2)  #256x13x13
        
        #Dropout:
        #self.dropout=nn.Dropout(p=probability)
        #self.dropout1 = nn.Dropout(p=0.1)
        #self.dropout2 = nn.Dropout(p=0.2)
        #self.dropout3 = nn.Dropout(p=0.3)
        #self.dropout4 = nn.Dropout(p=0.4)
        #self.dropout5 = nn.Dropout(p=0.5)
        #self.dropout6 = nn.Dropout(p=0.6)
        
        #Fully Connected layers:
        #self.dense=nn.Linear(in_features,out_features)
        #self.dense1=nn.Linear(256*13*13,1000)
        #self.dense2=nn.Linear(1000,1000)
        #self.dense3=nn.Linear(1000,68*2)
        
        #Batch Normalization:
        #self.bn=nn.BatchNorm2d(input_depth)
        #self.BN1=nn.BatchNorm2d(32)
        #self.BN2=nn.BatchNorm2d(64)
        #self.BN3=nn.BatchNorm2d(128)
        #self.BN4=nn.BatchNorm2d(256)
        #self.BN5 = nn.BatchNorm1d(1000)
       
        #self.conv1.weight = I.normal_(self.conv1.weight,mean=0.0, std=1.0)
        #self.conv2.weight = I.normal_(self.conv2.weight,mean=0.0, std=1.0)
        #self.conv3.weight = I.normal_(self.conv3.weight,mean=0.0, std=1.0)
        #self.dense1.weight= I.xavier_uniform_(self.dense1.weight,gain=1.0)
        #self.dense2.weight= I.xavier_uniform_(self.dense2.weight,gain=1.0)
        #self.dense3.weight= I.xavier_uniform_(self.dense3.weight,gain=1.0)
        
       
    
        #another model using 5x5 kernel size:
        
        #Filter details of convolution layers:
        #layer name :::: #of filters ::: filter shape
        #conv2d_1   :::     32       ::: (5,5)
        #conv2d_2   :::     64       ::: (5,5)
        #conv2d_3   :::     128      ::: (5,5)
        #conv2d_4   :::     256      ::: (5,5)
        
        #Convolution layers:
        #self.conv=nn.conv2d(input_depth,output_depth,kernel_size)
        self.conv1=nn.Conv2d(1,32,5)     #32x220x220
        self.conv2=nn.Conv2d(32,64,5)    #64x106x106
        self.conv3=nn.Conv2d(64,128,5)    #128x49x49
        self.conv4=nn.Conv2d(128,256,5)  #256x20x20
        
        #Maxpool layers:
        #self.pool=nn.MaxPool2d(kernel_size,stride)
        self.pool1 = nn.MaxPool2d(2, 2)  #32x110x110
        self.pool2 = nn.MaxPool2d(2, 2)  #64x53x53
        self.pool3 = nn.MaxPool2d(2, 2)  #128x24x24
        self.pool4 = nn.MaxPool2d(2, 2)  #256x10x10
        
        #Fully Connected layers:
        #self.dense=nn.Linear(in_features,out_features)
        self.dense1=nn.Linear(256*10*10,1000)
        #self.dense2=nn.Linear(1000,1000)
        self.dense2=nn.Linear(1000,68*2)
        
        #Dropout:
        #self.dropout=nn.Dropout(p=probability)
        #self.dropout1 = nn.Dropout(p=0.5)
        #self.dropout2 = nn.Dropout(p=0.5)
        #self.dropout3 = nn.Dropout(p=0.5)
        #self.dropout4 = nn.Dropout(p=0.5)
        #self.dropout5 = nn.Dropout(p=0.5)
        
        #Batch Normalization:
        #self.bn=nn.BatchNorm2d(input_depth)
        self.BN1=nn.BatchNorm2d(32)
        self.BN2=nn.BatchNorm2d(64)
        self.BN3=nn.BatchNorm2d(128)
        self.BN4=nn.BatchNorm2d(256)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool1(self.BN1(F.relu(self.conv1(x))))
        #x = self.dropout1(x)
        
        #x = self.pool2(self.BN2(F.relu(self.conv2(x))))
        #x = self.dropout2(x)
        
        #x = self.pool3(self.BN3(F.relu(self.conv3(x))))
        #x = self.dropout3(x)
        
        #x = self.pool4(self.BN4(F.relu(self.conv4(x))))
        #x = self.dropout4(x)
        
        # Flatten layer
        #x = x.view(x.size(0), -1)
        
        #x = F.elu(self.dense1(x))
        #x = self.dropout5(x)
        
        #x = F.elu(self.dense2(x))
        #x = self.dropout6(x)
        
        #x = self.dense3(x)
        
        ##The second model:
        x = self.pool1(self.BN1(F.relu(self.conv1(x))))
        #x = self.dropout1(x)
        
        x = self.pool2(self.BN2(F.relu(self.conv2(x))))
        #x = self.dropout2(x)
        
        x = self.pool3(self.BN3(F.relu(self.conv3(x))))
        #x = self.dropout3(x)
        
        x = self.pool4(self.BN4(F.relu(self.conv4(x))))
        #x = self.dropout4(x)
        
        # Flatten layer
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.dense1(x))
        #x = self.dropout5(x)
        
        x = self.dense2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
