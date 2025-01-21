import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import transforms as T
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt


def normalize(X):
    # Given a batch of examples 𝑋, normalize the examples to have zero mean
    # and unit variance along each feature.
    #
    # X := Tensor(float) of size (m,d) --- Batch of m examples of dimension d
    #
    # Return := Tensor of size (m,d) --- Normalized X along each feature


    return (X -  torch.mean(X,dim=0,keepdim=True))/torch.std(X,dim=0,keepdim= True)

def pca_fit(X,k):
    # Given a batch of examples 𝑋 and an integer 𝑘, calculate the top
    # 𝑘 PCA basis vectors.
    #
    # X := Tensor(float) of size (m,d) --- Batch of m examples of dimension d
    #
    # Return := Tensor of size (d, k) - The first `k` eigenvectors of the
    #       covariance matrix of `X`


    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    V = Vt.T[:, :k]

    return V



def pca_transform(X,V):
    # Given a batch of examples 𝑋 and 𝑘 PCA basis vectors, transform the
    # examples 𝑋 into the space spanned by the PCA basis vectors.
    #
    # X := Tensor (float) of size (m, d) - Batch of m examples of dimension d
    #
    # V := Tensor (float) of size (d, k) - First `k` principal components to
    #       use for the transformation
    #
    # Return := Tensor of size (m, k) - Transformed version of `X` projected
    #       onto the first `k` principal components represented by `V`

    # Fill in the rest
    return X @ V

def pca_reconstruction(X,V):
    # Given a batch of examples 𝑋 and 𝑘 PCA basis vectors, calculate the
    # best reconstruction of the examples 𝑋 from the PCA basis vectors
    #
    # X := Tensor (float) of size (m, d) - Batch of m examples of dimension d
    #
    # V := Tensor (float) of size (d, k) - The first `k` principal components
    #       used for the reconstruction
    #
    # Return := Tensor of size (m, d) - Reconstructed version of the original
    #       data `X`

    return (X@V) @ V.T


if __name__ == '__main__':
    # Plot reconstruction error as a function of k
    ks = [2**i for i in range(8)]
    losses = []
    for k in ks:
        V = pca_fit(normalize(X),k)
        pca_X = pca_reconstruction(X,V)
        pca_error = F.mse_loss(pca_X,X)
        losses.append(pca_error.item())
    plt.plot(ks,losses)