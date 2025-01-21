# packages for homework
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(30,1)
    def forward(self, X):
        out = self.linear(X)
        return out.squeeze()

    def fit_logistic_clf(X,y):
        clf = Logistic()
        opt = torch.optim.Adam(clf.parameters(), lr=0.1, weight_decay=1e2)
        loss = torch.nn.BCEWithLogitsLoss()
        for t in range(200):
            out = clf(X)
            opt.zero_grad()
            loss(out,(y>0).float()).backward()
            # if t % 50 == 0:
            #     print(loss(out,y.float()).item())
            opt.step()
        return clf

    def predict_logistic_clf(X, clf):
        return torch.sign(clf(X)).squeeze()
def boosting_fit(X, y, T, fit_logistic_clf, predict_logistic_clf):
    # X := Tensor(float) of size (m,d) -- Batch of m examples of demension d
    # y := Tensor(int) of size (m) -- the given vectors of labels of the examples
    # T := Maximum number of models to be implemented

    m = X.size(0)
    clfs = []
    while len(clfs) < T:
        # Calculate the weights for each sample mu. You may need to
        # divide this into the base case and the inductive case.

        ## ANSWER
        if len(clfs) == 0:
          mu = torch.ones(m)/m
        else:
          predictions = torch.zeros(m)
          for alpha, clf in clfs:
              predictions += alpha * predict_logistic_clf(X, clf)

          mu = torch.exp(-y * predictions)
          mu = mu / mu.sum()
        ## END ANSWER

        # Here, we draw samples according to mu and fit a weak classifier
        idx = torch.multinomial(mu, m, replacement=True)
        X0, y0 = X[idx], y[idx]

        clf = fit_logistic_clf(X0, y0)

        # Calculate the epsilon error term

        ## ANSWER
        predictions = predict_logistic_clf(X, clf)
        incorrect = (predictions != y)
        eps = torch.dot(mu, incorrect.float())
        ## END ANSWER

        if eps > 0.5:
            # In the unlikely even that gradient descent fails to
            # find a good classifier, we'll skip this one and try again
            continue

        # Calculate the alpha term here

        ## ANSWER
        alpha = 0.5 * torch.log2((1 - eps) / eps)
        ## END ANSWER

        clfs.append((alpha,clf))
    return clfs

def boosting_predict(X, clfs, predict_logistic_clf):
    # X := Tensor(float) of size (m,d) -- Batch of m examples of demension d
    # clfs := list of tuples of (float, logistic classifier) -- the list of boosted classifiers
    # Return := Tnesor(int) of size (m) -- the predicted labels of the dataset

    predictions = torch.zeros(X.size(0))

    for alpha, clf in clfs:
        predictions += alpha * predict_logistic_clf(X, clf)

    return torch.sign(predictions)


if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()
    data=train_test_split(cancer.data,cancer.target,test_size=0.2,random_state=123)

    torch.manual_seed(123)

    X,X_te,y,y_te = [torch.from_numpy(A) for A in data]
    X,X_te,y,y_te = X.float(), X_te.float(), torch.sign(y.long()-0.5), torch.sign(y_te.long()-0.5)


    logistic_clf = fit_logistic_clf(X,y)
    print("Logistic classifier accuracy:")
    print('Train accuracy: ', (predict_logistic_clf(X, logistic_clf) == y).float().mean().item())
    print('Test accuracy: ', (predict_logistic_clf(X_te, logistic_clf) == y_te).float().mean().item())

    boosting_clfs = boosting_fit(X,y, 10, fit_logistic_clf, predict_logistic_clf)
    print("Boosted logistic classifier accuracy:")
    print('Train accuracy: ', (boosting_predict(X, boosting_clfs, predict_logistic_clf) == y).float().mean().item())
    print('Test accuracy: ', (boosting_predict(X_te, boosting_clfs, predict_logistic_clf) == y_te).float().mean().item())