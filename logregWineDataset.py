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

class LogReg:
    def __init__(self):
        pass
    def logistic_loss(X, y, w):
    # Given a batch of samples and labels, and the weights of a logistic
    # classifier, compute the batched logistic loss.
    #
    # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
    #     of dimension d
    #
    # y := Tensor(int) of size (m,) --- This is a batch of m labels in {-1,1}
    #
    # w := Tensor(float) of size(d,) --- This is the weights of a logistic
    #     classifer.
    #
    # Return := Tensor of size (m,) --- This is the logistic loss for each
    #     example.


        vals = X.matmul(w)
        vals = -1* y* vals
        loss = torch.log(1+ torch.exp(vals))
        return loss



    def logistic_gradient(X, y, w):
        # Given a batch of samples and labels, compute the batched gradient of
        # the logistic loss.
        #
        # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
        #     of dimension d
        #
        # y := Tensor(int) of size (m,) --- This is a batch of m labels in {-1,1}
        #
        # w := Tensor(float) of size(d,) --- This is the weights of a logistic
        #     classifer.
        #
        # Return := Tensor of size (m,) --- This is the logistic loss for each
        #     example.

        bottom = -1/(1 +torch.exp(y*X.matmul(w)))
        return X *(y*bottom).unsqueeze(1)


    def optimize(X, y, niters=100):
        # Given a dataset of examples and labels, minimizes the logistic loss
        # using standard gradient descent.
        #
        # This optimizer is written for you, and you only need to implement the
        # logistic loss and gradient functions above.
        #
        # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
        #     of dimension d
        #
        # y := Tensor(int) of size (m,) --- This is a batch of m labels in {-1,1}
        #
        # Return := Tensor of size(d,) --- This is the fitted weights of a
        #     logistic regression model

        m,d = X.size()
        w = torch.zeros(d)
        print('Optimizing logistic function...')
        for i in range(niters):
            loss = logistic_loss(X,y,w).mean()
            grad = logistic_gradient(X,y,w).mean(0)
            w -= grad
            if i % 50 == 0:
                print(i, loss.item())
        print('Optimizing done.')
        return w

    def logistic_fit(X, y):
        # Given a dataset of examples and labels, fit the weights of the logistic
        # regression classifier using the provided loss function and optimizer
        #
        # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
        #     of dimension d
        #
        # y := Tensor(int) of size (m,) --- This is a batch of m labels in {-1,1}
        #
        # Return := Tensor of size (d,) --- This is the fitted weights of the
        #     logistic regression model

        # Fill in the rest. Hint -- call optimize :-).
        w= optimize(X,y)
        return w

    def logistic_predict(X, w):
        # Given a dataset of examples and fitted weights for a logistic regression
        # classifier, predict the class
        #
        # X := Tensor(float) of size(m,d) --- This is a batch of m examples of
        #    dimension d
        #
        # w := Tensor(float) of size (d,) --- This is the fitted weights of the
        #    logistic regression model
        #
        # Return := Tensor of size (m,) --- This is the predicted classes {-1,1}
        #    for each example
        #
        # Hint: Remember that logistic regression expects a label in {-1,1}, and
        # not {0,1}

        predictions = 1 / (1 + torch.exp(X.matmul(-w)))
        return torch.where(predictions < 0.5,-1,1)

if __name__ == '__main__':
    # Test your code on the wine dataset!
    # How does your solution compare to a random linear classifier?
    # Your solution should get around 75% accuracy on the test set.
    torch.manual_seed(42)
    logreg = LogReg()

    d = X_train.size(1)
    logistic_weights = {
        'zero': torch.zeros(d),
        'random': torch.randn(d),
        'fitted': logreg.logistic_fit(X_train, y_binary_train)
    }

    for k,w in logistic_weights.items():
        yp_binary_train = logreg.logistic_predict(X_train, w)
        acc_train = (yp_binary_train == y_binary_train).float().mean()

        print(f'Train accuracy [{k}]: {acc_train.item():.2f}')

        yp_binary_test = logreg.logistic_predict(X_test, w)
        acc_test = (yp_binary_test == y_binary_test).float().mean()

        print(f'Test accuracy [{k}]: {acc_test.item():.2f}')