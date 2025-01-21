import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn import datasets
import cvxpy as cp

from PIL import Image
from dill.source import getsource

import matplotlib
import matplotlib.pyplot as plt

%%capture
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names

class LinReg:
    def __init__(self):
        pass
    def regression_loss(X, y, w):
    # Given a batch of linear regression outputs and true labels, compute
    # the batch of squared error losses. This is *without* the ridge
    # regression penalty.
    #
    # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
    #     of dimension d
    #
    # y := Tensor(int) of size (m,) --- This is a batch of m real-valued labels
    #
    # w := Tensor(float) of size(d,) --- This is the weights of a linear
    #     classifer
    #
    # Return := Tensor of size (m,) --- This is the squared loss for each
    #     example

        loss = (X.matmul(w) - y)**2
        return loss

    def regression_fit(X, y, ridge_penalty=1.0):
        # Given a dataset of examples and labels, fit the weights of the linear
        # regression classifier using the provided loss function and optimizer
        #
        # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
        #     of dimension d
        #
        # y := Tensor(float) of size (m,) --- This is a batch of m real-valued
        #     labels
        #
        # ridge_penalty := float --- This is the parameter for ridge regression
        #
        # Return := Tensor of size (d,) --- This is the fitted weights of the
        #     linear regression model
        #
        # Fill in the rest

        n= y.size()[0]

        left= torch.linalg.inv(X.transpose(0,1).matmul(X) + ridge_penalty * n * torch.eye(X.size()[1]))
        right = X.transpose(0,1).matmul(y)

        return right.matmul(left)

        pass

    def regression_predict(X, w):
        # Given a dataset of examples and fitted weights for a linear regression
        # classifier, predict the label
        #
        # X := Tensor(float) of size(m,d) --- This is a batch of m examples of
        #    dimension d
        #
        # w := Tensor(float) of size (d,) --- This is the fitted weights of the
        #    linear regression model
        #
        # Return := Tensor of size (m,) --- This is the predicted real-valued labels
        #    for each example
        #
        # Fill in the rest
        return X.matmul(w)

if __name__ == '__main__':
    # Test your code on the wine dataset!
    # How does your solution compare to a random linear classifier?
    # Your solution should get an average squard error of about 8.6 test set.
    torch.manual_seed(42)
    linreg =LinReg()
    d = X_train.size(1)
    regression_weights = {
        'zero': torch.zeros(d),
        'random': torch.randn(d),
        'fitted': linreg.regression_fit(X_train, y_regression_train)
    }

    for k,w in regression_weights.items():
        yp_regression_train = linreg.regression_predict(X_train, w)
        squared_loss_train = linreg.regression_loss(X_train, y_regression_train, w).mean()

        print(f'Train accuracy [{k}]: {squared_loss_train.item():.2f}')

        yp_regression_test = linreg.regression_predict(X_test, w)
        squared_loss_test = linreg.regression_loss(X_test, y_regression_test, w).mean()

        print(f'Test accuracy [{k}]: {squared_loss_test.item():.2f}')