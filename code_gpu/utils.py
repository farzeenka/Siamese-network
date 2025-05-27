import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from imutils import build_montages
import cv2


def visualize(pairx, labels):
    pair1, pair2 = pairx
    pair1 = pair1[0]
    pair2 = pair2[0]
    plt.imsave("filename1.png", normalize(pair1))
    plt.imsave("filename2.png", normalize(pair2))


def pick_random_pairs(train_dataset):
    pairs = []
    labels = []
    for pair in train_dataset:
        prob = np.random.uniform(0, 1, 1)
        if prob > 0.9:
            pairs.append(pair[0])
            labels.append(pair[1])

    return pairs, labels


def buildMontages(train_dataset):
    images, labels = pick_random_pairs(train_dataset)
    unpack=[]
    for i in range(len(images)):
        unpack.append(images[i][0][0])
        unpack.append(images[i][1][0])
    unpack=[i.numpy() for i in unpack]

    return build_montages(unpack, (128, 128), (2, int(len(unpack)/2)))


def precision_recall(y_true, y_pred):
    pass


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def plotSimilarities(similarityl, labell):
    scatter_x = np.array(range(16))
    similarityl = np.array(similarityl)
    labell = np.array(labell)
    labell = [int(i[0]) for i in labell]

    cdict = {0: "red", 1: "blue"}
    labeldict = {0: "genuine-genuine", 1: "genuine-forged"}
    fig, ax = plt.subplots()
    for g in np.unique(labell):
        ix = np.where(labell == g)
        ax.scatter(scatter_x[ix], similarityl[ix], c=cdict[g], label=labeldict[g], s=100)
    ax.legend()
    plt.xlabel("IMAGES (TRAINING SET)")
    plt.ylabel("COSINE SIMILARITY")
    plt.savefig("similarity.png")