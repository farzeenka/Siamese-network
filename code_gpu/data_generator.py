"""
Script to prepare data for training and testing the network.
"""

import os
from natsort import natsorted
import random
from itertools import combinations, product
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from utils import visualize

# Parameters - using relative paths
forgeries_path = "../datasetF1/Invalid"  # Relative path to the 'Invalid' folder
real_path = "../datasetF1/Valid"         # Relative path to the 'Valid' folder
trainSplit = 18

def generateList(path):
    """
    Generate a list of image files in the given directory.

    Parameters:
        path (str): The path to the directory.

    Returns:
        list: A list of image file names.
    """
    imgs = os.listdir(path)
    imgs = [i for i in imgs if not i.endswith("Zone.Identifier")]
    imgs = natsorted(imgs, reverse=False)
    return imgs

def generatePairs(realSigs, forgedSigs, trainSplit):
    """
    Generates pairs of genuine and forged signatures for training and testing.

    Parameters:
        realSigs (list): List of real signature images.
        forgedSigs (list): List of forged signature images.
        trainSplit (int): Number of pairs to split for training.

    Returns:
        tuple: A tuple containing genuinePairs, genuineForgedPairs, realTest, and forgedTest.
    """
    realTrain = random.sample(realSigs, trainSplit)
    realTest = [i for i in realSigs if i not in realTrain]
    forgedTrain = random.sample(forgedSigs, trainSplit)
    forgedTest = [i for i in forgedSigs if i not in forgedTrain]

    genuinePairs = list(combinations(realTrain, 2))
    genuineForgedPairs = random.sample(
        list(product(realTrain, forgedTrain)), len(genuinePairs)
    )

    return genuinePairs, genuineForgedPairs, realTest, forgedTest

def createTestPairs(realTest, forgedTest):
    """
    Creates test pairs of genuine and forged signatures.

    Parameters:
        realTest (list): List of real signature images for test.
        forgedTest (list): List of forged signature images for test.

    Returns:
        tuple: A tuple containing genuineTestPairs and forgedTestPairs.
    """
    genuineTestPairs = list(combinations(realTest, 2))
    forgedTestPairs = random.sample(list(product(realTest, forgedTest)), 4)
    return genuineTestPairs, forgedTestPairs

def imageToArray(image_path):
    """
    Load an image from the given `image_path` and convert it to a numpy array.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image as a numpy array.
    """
    img = load_img(image_path, color_mode="rgb")
    img_array = img_to_array(img)
    return np.array(img_array)

def resizeImage(img_array, height=None, width=None):
    """
    Return the image array without resizing.
    """
    return img_array

import tensorflow as tf

def preprocessImages(image_array, target_height=120, target_width=160):

    """
    Perform preprocessing tasks such as resizing the image to a target height and width.

    Parameters:
        image_array (numpy.ndarray): The image to be preprocessed.
        target_height (int): The target height for resizing the image.
        target_width (int): The target width for resizing the image.

    Returns:
        numpy.ndarray: The preprocessed (resized) image array.
    """
    resized_image = tf.image.resize(image_array, [target_height, target_width])
    return resized_image.numpy()  # Convert the tensor back to a numpy array

def buildImgMatrix(real_path, forgeries_path, trainSplit):
    """
    Builds a matrix of images for training and testing.

    Args:
        real_path (str): The path to the directory containing real signature images.
        forgeries_path (str): The path to the directory containing forged signature images.
        trainSplit (int): The number of pairs to split for training.

    Returns:
        tuple: A tuple containing genuinePairsPix and genuineForgedPairsPix.
    """
    realSigs = generateList(real_path)
    forgedSigs = generateList(forgeries_path)

    genuinePairs, genuineForgedPairs, realTest, forgedTest = generatePairs(
        realSigs, forgedSigs, trainSplit
    )

    genuinePairsPix = []
    for i in range(len(genuinePairs)):
        genuinePairsPix.append(
            [
                preprocessImages(imageToArray(real_path + "/" + genuinePairs[i][0])),  
                preprocessImages(imageToArray(real_path + "/" + genuinePairs[i][1])),  
            ]
        )

    genuineForgedPairsPix = []
    for i in range(len(genuineForgedPairs)):
        genuineForgedPairsPix.append(
            [
                preprocessImages(imageToArray(real_path + "/" + genuineForgedPairs[i][0])),  
                preprocessImages(imageToArray(forgeries_path + "/" + genuineForgedPairs[i][1])),  
            ]
        )

    testPairsg, testPairsf = createTestPairs(realTest, forgedTest)

    testPairsPixg = []
    for i in range(len(testPairsg)):
        testPairsPixg.append(
            [
                preprocessImages(imageToArray(real_path + "/" + testPairsg[i][0])),  
                preprocessImages(imageToArray(real_path + "/" + testPairsg[i][1])),  
            ]
        )

    testPairsPixf = []
    for i in range(len(testPairsf)):
        testPairsPixf.append(
            [
                preprocessImages(imageToArray(real_path + "/" + testPairsf[i][0])),  
                preprocessImages(imageToArray(forgeries_path + "/" + testPairsf[i][1])),  
            ]
        )

    return genuinePairsPix, genuineForgedPairsPix, testPairsPixg, testPairsPixf

def createLabels(genuinePairs):
    """
    Creates labels for all pairs of signatures.

    Parameters:
        genuinePairs (list): A list of genuine signature pairs.

    Returns:
        list: A list of labels corresponding to all pairs.
    """
    labels = []
    for i in range(len(genuinePairs)):
        labels.append(0)

    for i in range(len(genuinePairs)):
        labels.append(1)
    return labels

def createTestLabels():
    """
    Creates labels for test pairs of signatures.

    Returns:
        list: A list of labels corresponding to all pairs.
    """
    labels = []
    labels.append(0)

    for i in range(4):
        labels.append(1)
    
    return labels

def prepareTfDataset(real_path, forgeries_path, trainSplit):
    """
    Prepares a TensorFlow dataset for training by building image matrices, batching, and prefetching.

    Args:
        real_path (str): The path to the directory containing real signature images.
        forgeries_path (str): The path to the directory containing forged signature images.
        trainSplit (int): The number of pairs to split for training.

    Returns:
        tf.data.Dataset: A TensorFlow dataset prepared for training.
    """
    genuinePairsPix, genuineForgedPairsPix, testPairsPixg, testPairsPixf = buildImgMatrix(real_path, forgeries_path, trainSplit)
    allPairsPix = genuinePairsPix + genuineForgedPairsPix

    pair1 = []
    pair2 = []
    for i in range(len(allPairsPix)):
        pair1.append(allPairsPix[i][0])
        pair2.append(allPairsPix[i][1])

    labels = createLabels(genuinePairsPix)
    labels = tf.cast(labels, tf.float32)

    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    pixels_dataset = tf.data.Dataset.from_tensor_slices((pair1, pair2))
    dataset = tf.data.Dataset.zip((pixels_dataset, labels_dataset))
    dataset = dataset.batch(3)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def prepareTfDatasetTest(real_path, forgeries_path, trainSplit):
    """
    Prepares a TensorFlow dataset for testing by building image matrices, batching, and prefetching.

    Args:
        real_path (str): The path to the directory containing real signature images.
        forgeries_path (str): The path to the directory containing forged signature images.

    Returns:
        tf.data.Dataset: A TensorFlow dataset prepared for testing.
    """
    genuinePairsPix, genuineForgedPairsPix, testPairsPixg, testPairsPixf = buildImgMatrix(real_path, forgeries_path, trainSplit)
    allTestPix = testPairsPixg + testPairsPixf

    pair1 = []
    pair2 = []
    for i in range(len(allTestPix)):
        pair1.append(allTestPix[i][0])
        pair2.append(allTestPix[i][1])

    labels = createTestLabels()
    labels = tf.cast(labels, tf.float32)

    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    pixels_dataset = tf.data.Dataset.from_tensor_slices((pair1, pair2))
    dataset = tf.data.Dataset.zip((pixels_dataset, labels_dataset))
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# testing
if __name__ == "__main__":
    dataset = prepareTfDatasetTest("../datasetF1/Invalid", "../datasetF1/Valid", 18)
    """
    Add test cases to assert dataset is as expected and requirements are met
    """

    c = 0
    for i in dataset:
        if c < 1:
            print(i)
        else:
            break
        c += 1
