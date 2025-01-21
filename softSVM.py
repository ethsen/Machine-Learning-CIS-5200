import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn import datasets
import cvxpy as cp

from PIL import Image
from dill.source import getsource

import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize._lsq.dogbox import LinearOperator
from sklearn import datasets

class SoftSVM:
    def __init__(self, ndims):
        # Here, we initialize the parameters of your soft-SVM model for binary
        # classification. Don't change the weight and bias variables as the
        # autograder will assume that these exist.
        # ndims := integer -- number of dimensions
        # no return type
        import torch

        self.weight = torch.zeros(ndims)
        self.bias = torch.zeros(1)
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def objective(self, X, y, l2_reg):
        # Calculate the objective of your soft-SVM model
        # X := Tensor of size (m,d) -- the input features of m examples with d dimensions
        # y := Tensor of size (m) -- the labels for each example in X
        # l2_reg := float -- L2 regularization penalty
        # Returns a scalar tensor (zero dimensional tensor) -- the loss for the model

        modelPerformance = y* (X.matmul(self.weight) +self.bias)
        xi = torch.where(modelPerformance < 1,1-modelPerformance,0)

        loss = torch.mean(xi) + ( l2_reg*self.weight.matmul(self.weight))

        return loss


    def gradient(self, X, y, l2_reg):
        # Calculate the gradient of your soft-SVM model
        # X := Tensor of size (m,d) -- the input features of m examples with d dimensions
        # y := Tensor of size (m) -- the labels for each example in X
        # l2_reg := float -- L2 regularization penalty
        # Return Tuple (Tensor, Tensor) -- the tensors corresponds to the weight
        # and bias parameters respectively
        # Fill in the rest
        modelPerformance = y* (X.matmul(self.weight) +self.bias)
        mask = (modelPerformance<1).float()

        weightGrad =  -X.t().matmul(y*mask)/X.size()[0] + 2*l2_reg*self.weight
        biasGrad = (-y*mask).mean()

        return weightGrad, biasGrad

    def optimize(self, X, y, l2_reg,learningRate = 0.2, epochs = 100):
        # Calculate the gradient of your soft-SVM model
        # X := Tensor of size (m,d) -- the input features of m examples with d dimensions
        # y := Tensor of size (m) -- the labels for each example in X
        # l2_reg := float -- L2 regularization penalty

        for epoch in range(epochs):
            weightGrad, biasGrad = self.gradient(X,y,l2_reg)
            with torch.no_grad():
              self.weight -= learningRate*weightGrad
              self.bias -= learningRate*biasGrad
            loss = self.objective(X,y,l2_reg)


    def predict(self, X):
        # Given an X, make a prediction with the SVM
        # X := Tensor of size (m,d) -- features of m examples with d dimensions
        # Return a tensor of size (m) -- the prediction labels on the dataset X

        # Fill in the rest
        return torch.sign(X.matmul(self.weight) +self.bias)
    

if __name__ == '__main__':

    #Load dataset
    cancer = datasets.load_breast_cancer()
    X,y = torch.from_numpy(cancer['data']), torch.from_numpy(cancer['target'])
    mu,sigma = X.mean(0,keepdim=True), X.std(0,keepdim=True)
    X,y = ((X-mu)/sigma).float(),(y - 0.5).sign() # prepare data
    l2_reg = 0.1
    print(X.size(), y.size())

    # Optimize the soft-SVM with gradient descent
    clf = SoftSVM(X.size(1))
    clf.optimize(X,y,l2_reg)
    print("\nSoft SVM objective: ")
    print(clf.objective(X,y,l2_reg).item())
    print("\nSoft SVM accuracy: ")
    (clf.predict(X) == y).float().mean().item()