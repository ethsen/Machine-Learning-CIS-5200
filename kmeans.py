import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import transforms as T
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

def kmeans_init(X, k):
    samples = X.shape[0]
    centers = []

    firstCenter = torch.randint(0, samples, (1,)).item()
    first_center_tensor = X[firstCenter].reshape(1, -1)
    centers = [first_center_tensor]

    for i in range(1, k):
        center_tensor = torch.cat(centers, dim=0)

        dists = torch.cdist(X, center_tensor, p=2)
        minDist = torch.min(dists, dim=1).values

        prob_dist = (minDist**2)/torch.sum(minDist**2)
        idx = torch.multinomial(prob_dist, 1)

        new_center = X[idx].reshape(1, -1)
        centers.append(new_center)

    return torch.cat(centers, dim=0)


def kmeans_assign_clusters(X,mus):
    # docs

    distances = torch.cdist(X, mus, p=2)

    clusters = torch.argmin(distances, dim=1)

    return clusters

def kmeans_update_centroids(X,clusters):
    # docs

    k = len(torch.unique(clusters))
    new_centroids = []

    for i in range(k):
        mask = clusters == i
        cluster_points = X[mask]

        if len(cluster_points) > 0:
            centroid = torch.mean(cluster_points, dim=0)
        else:
            random_idx = torch.randint(len(X), (1,)).item()
            centroid = X[random_idx]

        new_centroids.append(centroid)

    return torch.stack(new_centroids)

def kmeans_stopping_criteria(clusters, prev_clusters):
    # docs

    if prev_clusters is None:
        return False

    return torch.all(clusters == prev_clusters)

if __name__ == '__main__':
    %%capture
    !wget https://archive.ics.uci.edu/ml/machine-learning-databases/00517/data.zip
    !unzip -o data.zip
    t = T.Compose([T.Resize(64),T.CenterCrop(32)])

    data_path = 'data'
    filenames = [name for name in os.listdir(data_path) if os.path.splitext(name)[-1] == '.jpg']
    imgs =[torchvision.io.read_image(os.path.join(data_path,f)) for f in filenames]
    imgs = torch.stack([t(im).float()/255 for im in imgs]).mean(1)
    X = imgs.view(imgs.size(0),-1)
    torch.manual_seed(42)
    k = 16
    niters = 500
    mus = kmeans_init(X,k)
    clusters = torch.randint(0,k,(X.size(0),))

    for t in range(niters):
        prev_clusters = clusters
        clusters = kmeans_assign_clusters(X,mus)
        mus = kmeans_update_centroids(X,clusters)

        if kmeans_stopping_criteria(clusters, prev_clusters):
            break
    print(f"Ended after {t} iterations.")

