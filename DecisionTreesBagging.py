# packages for homework
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

class DecisionTree:
  def entropy(self, y):
      # Calculate the entropy of a given vector of labels in the `entropy` function.
      #
      # y := Tensor(int) of size (m) -- the given vector of labels
      # Return a float that is your calculated entropy

      # Fill in the rest
      if len(y) == 0:
        return 0.0

      labels, counts = torch.unique(y, return_counts=True)
      probabilities = counts.float() / y.size(0)

      entropy = -torch.sum(probabilities * torch.log2(probabilities))

      return entropy.item()

  def find_split(self, node, X, y, k=4):
      # Find the best split over all possible splits that minimizes entropy.
      #
      # node := Map(string: value) -- the tree represented as a Map, the key will take four
      #   different string: 'label', 'split','left','right' (See implementation below)
      #   'label': a label node with value as the mode of the labels
      #   'split': the best split node with value as a tuple of feature id and threshold
      #   'left','right': the left and right branch with value as the label node
      # X := Tensor of size (m,d) -- Batch of m examples of demension d
      # y := Tensor(int) of size (m) -- the given vectors of labels of the examples
      # k := int -- the number of classes, with default value as 4
      # Return := tuple of (int, float) -- the feature id and threshold of the best split

      m = y.size(0)
      best_H, best_split = 999, None
      features = torch.randint(0, 4,(k,))

      for feature_idx in features:
          for threshold in torch.arange(0.15,7.9,0.1):
              idx = X[:,feature_idx] > threshold
              ####################################
              # THIS LINE BELOW WILL REMOVE UNKNOWN OPCODE
              ####################################
              if idx.sum() == 0 or idx.sum() == idx.size(0):
                  continue

              m_left = idx.sum()
              m_right = (~idx).sum()

              H_left = self.entropy(y[idx])
              H_right = self.entropy(y[~idx])
              ## ANSWER
              split_H = len(y[idx])/len(y) * H_left + len(y[~idx])/len(y) * H_right
              ## END ANSWER

              ####################################
              # THIS LINE BELOW WILL REMOVE UNKNOWN OPCODE
              ####################################
              if split_H < best_H or best_split == None:
                  best_H, best_split = split_H, (feature_idx, threshold)
      return best_split

  def expand_node(self, node, X, y, max_depth=0, k=4):
      # Completing the recursive call for building the decision tree
      # node := Map(string: value) -- the tree represented as a Map, the key will take four
      #   different string: 'label', 'split','left','right' (See implementation below)
      #   'label': a label node with value as the mode of the labels
      #   'split': the best split node with value as a tuple of feature id and threshold
      #   'left','right': the left and right branch with value as the label node
      # X := Tensor of size (m,d) -- Batch of m examples of demension d
      # y := Tensor(int) of size (m) -- the given vectors of labels of the examples
      # max_depth := int == the deepest level of the the decision tree
      # k := int -- the number of classes, with default value as 4
      # Return := tuple of (int, float) -- the feature id and threshold of the best split
      #

      H = self.entropy(y)
      if H == 0 or max_depth == 0:
          return

      best_split = self.find_split(node, X, y, k=k)

      ####################################
      # THIS LINE BELOW WILL REMOVE UNKNOWN OPCODE
      ####################################
      if best_split == None:
          return

      idx = X[:,best_split[0]] > best_split[1]
      X_left, y_left = X[idx], y[idx]
      X_right, y_right = X[~idx], y[~idx]

      del node['label']
      node['split'] = best_split
      node['left'] = { 'label': y_left.mode().values }
      node['right'] = { 'label': y_right.mode().values }

      # Fill in the following two lines to recursively build the rest of the
      # decision tree
      # self.expand_node(...)
      # self.expand_node(...)
      ## ANSWER
      self.expand_node(node['left'], X_left, y_left, max_depth=max_depth - 1, k=k)
      self.expand_node(node['right'], X_right, y_right, max_depth=max_depth - 1, k=k)

      ## END ANSWER

      return

  def predict_one(self, node, x):
      # Makes a prediction for a single example.
      # node := Map(string: value) -- the tree represented as a Map, the key will take four
      #   different string: 'label', 'split','left','right' (See implementation below)
      #   'label': a label node with value as the mode of the labels
      #   'split': the best split node with value as a tuple of feature id and threshold
      #   'left','right': the left and right branch with value as the label node
      # x := Tensor(float) of size(d,) -- the single example in a batch
      # Fill in the rest

      if 'label' in node:
        return node['label']

      if x[node['split'][0]] > node['split'][1]:
        return self.predict_one(node['left'], x)
      else:
        return self.predict_one(node['right'], x)

def fit_decision_tree(X,y, k=4):
    # The function will fit data with decision tree with the expand_node method implemented above

    root = { 'label': y.mode().values }
    dt = DecisionTree()
    dt.expand_node(root, X, y, max_depth=10, k=k)
    return root

def predict(node, X):
    # return the predict result of the entire batch of examples using the predict_one function above.
    dt = DecisionTree()
    return torch.stack([dt.predict_one(node, x) for x in X])

def bootstrap(X,y):
    # Draw a random bootstrap dataset from the given dataset.
    #
    # X := Tensor(float) of size (m,d) -- Batch of m examples of demension d
    # y := Tensor(int) of size (m) -- the given vectors of labels of the examples
    #
    # Return := Tuple of (Tensor(float) of size (m,d),Tensor(int) of size(m,)) -- the random bootstrap
    #       dataset of X and its correcting lable Y
    # Fill in the rest

    idx = torch.randint(0,y.size(0), (y.size(0),))

    return X[idx] ,y[idx]

def random_forest_fit(X, y, m, k, clf, bootstrap):
    # Train a random forest that fits the data.
    # X := Tensor(float) of size (n,d) -- Batch of n examples of demension d
    # y := Tensor(int) of size (n) -- the given vectors of labels of the examples
    # m := int -- number of trees in the random forest
    # k := int -- number of classes of the features
    # clf := function -- the decision tree model that the data will be trained on
    # bootstrap := function -- the function to use for bootstrapping (pass in "bootstrap")
    #
    # Return := the random forest generated from the training datasets
    # Fill in the rest
    forrest = []
    for i in range(m):
      xBootstrap, yBootstrap = bootstrap(X,y)
      tree = clf(xBootstrap,yBootstrap,k)
      forrest.append(tree)

    return forrest

def random_forest_predict(X, clfs, predict):
    # Implement `predict_forest_fit` to make predictions given a random forest.
    # X := Tensor(float) of size (m,d) -- Batch of m examples of demension d
    # clfs := list of functions -- the random forest
    # predict := function that predicts (will default to your "predict" function)
    # Return := Tensor(int) of size (m,) -- the predicted label from the random forest
    # Fill in the rest
    predictions = []
    for tree in clfs:
      predictions.append(predict(tree, X))
    vals, idx = torch.mode(torch.stack(predictions),dim = 0)
    return vals


if __name__ == '__main__':
    iris = datasets.load_iris()
    data=train_test_split(iris.data,iris.target,test_size=0.5,random_state=123)

    X,X_te,y,y_te = [torch.from_numpy(A) for A in data]
    X,X_te,y,y_te = X.float(), X_te.float(), y.long(), y_te.float()

    
    torch.manual_seed(42)
    RF = random_forest_fit(X,y,50,2, clf=fit_decision_tree, bootstrap=bootstrap)

    print('Train accuracy: ', (random_forest_predict(X, RF, predict) == y).float().mean().item())
    print('Test accuracy: ', (random_forest_predict(X_te, RF, predict) == y_te).float().mean().item())