import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import transforms as T
from torch.distributions.multivariate_normal import MultivariateNormal

import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # Start of the implementation
        self.linear = nn.Linear(in_dim, out_dim)
        self.reLu= nn.ReLU()
        # End of the implementation

    def forward(self, x):
        # Start of the implementation
        x = self.linear(x)
        x = self.reLu(x)
        return x
        # End of the implementation

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # Start of the implementation
        self.linear =  nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        # End of the implementation

    def forward(self, x):
        # Start of the implementation
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
        # End of the implementation

def train_autoencoder(encoder, decoder, optimizer, train_loader, epoch):
    encoder.train()
    decoder.train()

    # We use the mean squared error loss
    criterion = nn.MSELoss()

    # Start of the implementation
    totalLoss = 0
    for id, (data,_) in enumerate(train_loader):
      optimizer.zero_grad()
      encodedData =encoder(data)
      decodedData = decoder(encodedData)
      loss = criterion(decodedData, data)
      loss.backward()
      optimizer.step()
      totalLoss += loss.item()
    # End of the implementation



def visualize_autoencoder(encoder, decoder):
    idx = torch.randint(high=10000, size=(1,)).item()

    print("image index {:d}".format(idx))

    img = test_set.data[idx]
    x = img.flatten().float()

    fig, axes = plt.subplots(1, 2, figsize=(4, 8))

    axes[0].imshow(x.reshape(28, 28), cmap='gray')
    axes[0].set_title('original image')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(decoder(encoder(x)).detach().numpy().reshape(28, 28), cmap='gray')
    axes[1].set_title('reconstructed')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.show()


if __name__ == '__main__':
    transformation = transforms.Compose([transforms.ToTensor(), torch.flatten])

    train_set = MNIST('./data', train=True, download=True, transform=transformation)
    test_set = MNIST('./data', train=False, download=True, transform=transformation)

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=0)

    encoder = Encoder(784, 50)
    decoder = Decoder(50, 784)

    visualize_autoencoder(encoder, decoder)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    train_autoencoder(encoder, decoder, optimizer, train_loader, epoch=10)
    visualize_autoencoder(encoder, decoder)