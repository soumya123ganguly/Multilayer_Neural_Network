
import copy
import util
from neuralnet import *
from tqdm import tqdm

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """

    # Read in the esssential configs
    epochs = config["epochs"]
    early_stopping = config["early_stop"]
    max_patience = config["early_stop_epoch"]
    patience = 0
    min_valid_loss = 1e100

    trainEpochLoss = []
    trainEpochAccuracy = []
    validEpochLoss = []
    validEpochAccuracies = []
    for epoch in tqdm(range(epochs)):
        num_train_samples = len(x_train)
        num_valid_samples = len(x_valid)
        batch_size = config["batch_size"]
        train_mb_itr = util.generate_minibatches((x_train, y_train), 
                                                 batch_size=batch_size)
        valid_mb_itr = util.generate_minibatches((x_valid, y_valid), 
                                                 batch_size=batch_size)
        train_loss = 0
        train_accuracy = 0
        for _ in range(num_train_samples//batch_size):
            x_train_mb, y_train_mb = next(train_mb_itr)
            loss = model.forward(x_train_mb, targets=y_train_mb)
            yh_train_mb = model.y
            model.backward()
            train_loss += loss
            train_accuracy += util.calculateCorrect(yh_train_mb, y_train_mb)
        train_loss /= (num_train_samples//batch_size)
        train_accuracy /= (num_train_samples//batch_size)

        valid_loss = 0
        valid_accuracy = 0
        for _ in range(num_valid_samples//batch_size):
            x_valid_mb, y_valid_mb = next(valid_mb_itr)
            loss = model.forward(x_valid_mb, targets=y_valid_mb)
            yh_valid_mb = model.y
            valid_loss += loss
            valid_accuracy += util.calculateCorrect(yh_valid_mb, y_valid_mb)
        valid_loss /= (num_valid_samples//batch_size)
        valid_accuracy /= (num_valid_samples//batch_size)
        
        # Update train and validation losses and accuracies
        trainEpochLoss.append(train_loss)
        trainEpochAccuracy.append(train_accuracy)
        validEpochLoss.append(valid_loss)
        validEpochAccuracies.append(valid_accuracy)

        # Early Stopping
        if early_stopping:
            if valid_loss > min_valid_loss:
                patience += 1
                if patience == max_patience:
                    util.plots(trainEpochLoss, 
                               trainEpochAccuracy, 
                               validEpochLoss, 
                               validEpochAccuracies, 
                               epoch)
                    return model
            else:
                min_valid_loss = valid_loss
                patience = 0

    # Save plots
    util.plots(trainEpochLoss, 
               trainEpochAccuracy, 
               validEpochLoss, 
               validEpochAccuracies, 
               epoch)

    return model

#This is the test method
def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    num_test_samples = len(X_test)
    batch_size = 64
    test_mb_itr = util.generate_minibatches((X_test, y_test), 
                                                batch_size=batch_size)
    test_loss = 0
    test_accuracy = 0
    for _ in range(num_test_samples//batch_size):
        x_test_mb, y_test_mb = next(test_mb_itr)
        loss = model.forward(x_test_mb, targets=y_test_mb)
        yh_test_mb = model.y
        test_loss += loss
        test_accuracy += util.calculateCorrect(yh_test_mb, y_test_mb)
    test_loss /= (num_test_samples//batch_size)
    test_accuracy /= (num_test_samples//batch_size)
    return test_accuracy, test_loss


