import copy
import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants


def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)



def normalize_data(inp):
    """
    TODO
    Normalizes inputs (on per channel basis of every image) here to have 0 mean and unit variance.
    This will require reshaping to seprate the channels and then undoing it while returning

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions

    returns:
        normalized inp: N X d 2D array

    """
    channel_size = inp.shape[1]//3
    def normalize_channel(inp):
        """
            Normalize a channel.
        """
        mean = inp.mean()
        std = inp.std()
        return (inp-mean)/std
    # Normalize each channel seperately
    inp_normalize_r = normalize_channel(inp[:, :channel_size])
    inp_normalize_g = normalize_channel(inp[:, channel_size:2*channel_size])
    inp_normalize_b = normalize_channel(inp[:, 2*channel_size:])
    # Stack all the channels together
    inp_normalized_images = np.concatenate((inp_normalize_r, 
                                            inp_normalize_g, 
                                            inp_normalize_b), axis=1)
    return inp_normalized_images

def one_hot_encoding(labels, num_classes=20):
    """
    TODO
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (20/100 for CIFAR-100)

    returns:
        oneHot : N X num_classes 2D array

    """
    num_samples = len(labels)
    z = np.zeros((num_samples, num_classes))
    z[np.arange(num_samples), labels.reshape(-1)-1] = 1
    return z

def generate_minibatches(dataset, batch_size=64):
    """
        Generates minibatches of the dataset

        args:
            dataset : 2D Array N (examples) X d (dimensions)
            batch_size: mini batch size. Default value=64

        yields:
            (X,y) tuple of size=batch_size

        """

    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def calculateCorrect(y,t):  #Feel free to use this function to return accuracy instead of number of correct predictions
    """
    TODO
    Calculates the number of correct predictions

    args:
        y: Predicted Probabilities
        t: Labels in one hot encoding

    returns:
        the number of correct predictions
    """
    predicted_labels = np.argmax(y, axis=1)
    target_labels = np.argmax(t, axis=1)
    return (predicted_labels == target_labels).mean()*100


def append_bias(X):
    """
    TODO
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    num_samples = len(X)
    return np.hstack((X, np.ones((num_samples,1))))



def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):

    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. earlyStop=-1 means the last epoch was the best one
    """

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(constants.saveLocation+"loss.eps")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(constants.saveLocation+"accuarcy.eps")
    plt.show()

    #Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.saveLocation+"trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.saveLocation+"valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(constants.saveLocation+"trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(constants.saveLocation+"valEpochAccuracy.csv")



def createTrainValSplit(x_train,y_train):

    """
    TODO
    Creates the train-validation split (80-20 split for train-val). Please shuffle the data before creating the train-val split.
    """
    n = len(x_train)
    idx = np.random.permutation(n)
    x_train = x_train[idx]
    y_train = y_train[idx]
    return (x_train[:4*n//5], y_train[:4*n//5], x_train[4*n//5:], y_train[4*n//5:])


def load_data(path, num_classes=20):
    """
    Loads, splits our dataset- CIFAR-100 into train, val and test sets and normalizes them

    args:
        path: Path to cifar-100 dataset
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, constants.cifar100_directory)

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    images_dict = unpickle(os.path.join(cifar_path, "train"))
    data = images_dict[b'data']
    # Set train labels
    if num_classes == 100:
        label = images_dict[b'fine_labels']
    else:
        label = images_dict[b'coarse_labels']
    train_labels.extend(label)
    train_images.extend(data)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels).reshape((len(train_labels),-1))
    # Create train and validation split
    train_images, train_labels, val_images, val_labels = createTrainValSplit(train_images,train_labels)

    # Normalize train images and generate one hot encoding labels
    train_normalized_images = normalize_data(train_images)
    train_one_hot_labels = one_hot_encoding(train_labels, num_classes)

    # Normalize validation images and generate one hot encoding labels
    val_normalized_images = normalize_data(val_images)
    val_one_hot_labels = one_hot_encoding(val_labels, num_classes)

    test_images_dict = unpickle(os.path.join(cifar_path, "test"))
    test_data = test_images_dict[b'data']
    # Set test labels
    if num_classes == 100:
        test_labels = test_images_dict[b'fine_labels']
    else:
        test_labels = test_images_dict[b'coarse_labels']
    test_images = np.array(test_data)
    test_labels = np.array(test_labels).reshape((len(test_labels), -1))
    # Normalize test images and generate one hot encoding labels
    test_normalized_images= normalize_data(test_images)
    test_one_hot_labels = one_hot_encoding(test_labels, num_classes)
    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels
