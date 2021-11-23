# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:42:50 2021

@author: vitfv
"""

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as skm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
mat = scipy.io.loadmat('ex3data1.mat')#MNIST

X = mat['X']
y = mat['y']
y = OneHotEncoder(sparse=False).fit_transform(y)
plt.imshow(X[2].reshape([20,20]))

class Neuron_layer():
    
    def __init__(self, weights, alpha = 0.01,  bias=None, activation = lambda x : x, activation_deriv = lambda x : 1):
        self.weights = weights
        if (bias is not None):
            self.bias = bias
        else:
            self.bias=None
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.input = None
        self.output = None
        self.grad = 0
        self.alpha=alpha
    
    def feedforward(self,inputs):
        self.input=inputs.reshape(-1,1)
        result = self.activation(np.dot(self.weights.T,inputs)+self.bias)
        self.output = result.reshape(-1,1)
        return result
    def backpropagate(self,error):
        self.grad = np.dot(self.input, error.T)
        layer_error = np.dot(self.weights,error) * self.activation_deriv(self.input)
        self.weights += self.alpha* self.grad
        if(self.bias is not None):
            self.bias += self.alpha*np.dot(1,error.T.reshape(-1))
        return layer_error
        
    
    
class NeuralNet():
    def __init__(self,layers,sizes,loss,metric,activations,activations_deriv,alpha,biases):
        self.layers = []
        np.random.seed(42)
        for i in range(1,layers):
            weights = np.random.uniform(low=-1.0, high=1.0, size=(sizes[i-1],sizes[i]))
            biases =  np.random.uniform(low=-1.0, high=1.0, size=sizes[i])
            self.layers.append(Neuron_layer(weights,
                                            alpha,
                                            biases,
                                            activations[i],
                                            activations_deriv[i]))
            self.loss = loss
            self.metric = metric
    
    def fit(self,X,y,epochs,batch_size=1):
        print('Start training')
        for i in range(epochs):
            print(f'Epoch {i}')
            ypr = np.zeros([y.shape[0],y.shape[1]])
            for j in range(X.shape[0]):
                result = X[j]
                for k in self.layers:
                    result = k.feedforward(result)
                ypr[j] = result
               # metric = self.metric(y[j],result)
                difference = (y[j] - result).reshape(-1,1)
                for k in reversed(range(len(self.layers))):
                    difference = self.layers[k].backpropagate(difference)
            loss = self.loss(y,ypr)
            print(f'loss {loss}')
        print('Training complete!')
        print(f'Final loss {loss}')
        
    def predict_proba(self,X):
        probas = np.zeros([X.shape[0],self.layers[-1].output.shape[0]]) 
        for j in range(len(X)):
            result = X[j]
            for i in self.layers:
                result = i.feedforward(result)
            probas[j]=result
        return probas
    def predict(self,X,threshold = 'max'):
        return np.argmax(self.predict_proba(X),axis=1)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return x*(1.0-x)

activations = [sigmoid,sigmoid,sigmoid, sigmoid]
activations_deriv = [sigmoid_prime,sigmoid_prime,sigmoid_prime,sigmoid_prime]

weights = scipy.io.loadmat('ex3weights.mat')#Precomputed weights

theta1 = weights['Theta1']
theta2 = weights['Theta2']

X_train,X_test,y_train,y_test = train_test_split(X,y, shuffle=True)

#Setting weights with precomputed ones
nn_pretrained = NeuralNet(3,[400,25,10],skm.log_loss,skm.accuracy_score,activations[:3],activations_deriv[:3],0.01,[False,True,True])
nn_pretrained.layers[0].weights = theta1.T[1:,:]
nn_pretrained.layers[1].weights = theta2.T[1:,:]
nn_pretrained.layers[0].bias = theta1.T[:1,:].reshape(-1)
nn_pretrained.layers[1].bias = theta2.T[:1,:].reshape(-1)
nn_pretrained.layers[-1].output = np.zeros([10,1])
predictions_pretrained = nn_pretrained.predict(X_test)
real_values = np.argmax(y_test,axis=1)

nn = NeuralNet(3,[400,25,10],skm.log_loss,skm.accuracy_score,activations[:3],activations_deriv[:3],0.1,[False,True,True])
nn.fit(X_train,y_train,100)

predictions = nn.predict(X_test)

accuracy_my_net = skm.accuracy_score(real_values,predictions)
accuracy_pretrained = skm.accuracy_score(real_values,predictions_pretrained)

print(f'Accuracy для моей нейронной сети - {accuracy_my_net}')
print(f'Accuracy для загруженных весов - {accuracy_pretrained}')


