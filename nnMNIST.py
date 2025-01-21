import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from sklearn import datasets
import pandas as pd

import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        # Start of the implementation
        self.linear1 = torch.nn.Linear(in_dim,hidden,bias = True)
        self.linear2 = torch.nn.Linear(hidden, out_dim, bias = True)
        # End of the implementation

    def forward(self, x):
        # Start of the implementation
        h = torch.nn.functional.relu(self.linear1(x))
        out = self.linear2(h)
        return out
        # End of the implementation

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Evaluate the model accuracy on the training/test set by looping over the data loader.

        Args:
            self: this neural network
            loader: a PyTorch data loader

        Returns:
            (a float scalar) the accuracy on the training/test set
        """
        self.eval()

        correct = 0.
        total = 0.

        # Start of the implementation
        with torch.no_grad():
          for inputs, targets in loader:
            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        return correct / total

    def train_model(self, optimizer, train_loader, test_loader, epoch=20):
        """
        Train the neural network.

        Args:
            self: this neural network
            train_loader: a PyTorch data loader on the training set
            test_loader: a PyTorch data loader on the test set
            epoch: the number of epochs

        Returns:
            None, the model is updated in-place
        """
        self.train()

        # we use the cross entropy loss for classification -- feel free to read the docs for this class: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        criterion = nn.CrossEntropyLoss()
        # Start of the implementation
        for i in range(epoch):
          for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs =self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()



        # End of the implementation


def visualize(model):
    idx = torch.randint(high=10000, size=(1,)).item()

    print("image index {:d}".format(idx))

    img = test_set.data[idx]
    x = img.flatten().float()

    prediction = model(x).argmax(dim=-1)
    print("prediction {:d}".format(prediction))

    plt.figure(figsize=(2, 2))
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    plt.show()

if __name__ == '__main__':
    transformation = transforms.Compose([transforms.ToTensor(), torch.flatten])

    train_set = MNIST('./data', train=True, download=True, transform=transformation)
    test_set = MNIST('./data', train=False, download=True, transform=transformation)

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=0)

    model = Network(784, 100, 10)

    acc_train_set = model.evaluate(train_loader)
    acc_test_set = model.evaluate(test_loader)

    print("training set acc {:f}, test set acc {:f}".format(acc_train_set, acc_test_set))


    # the optimizer is stochastic gradient descent with Nesterov's momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, nesterov=True)

    model.train_model(optimizer, train_loader, test_loader, epoch=20)

    acc_train_set = model.evaluate(train_loader)
    acc_test_set = model.evaluate(test_loader)

    print("training set acc {:f}, test set acc {:f}".format(acc_train_set, acc_test_set))