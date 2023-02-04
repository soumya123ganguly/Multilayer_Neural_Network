
import copy
import util
from neuralnet import *

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
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    for epoch in range(epochs):
        print(epoch)
        num_train_samples = len(x_train)
        num_valid_samples = len(x_valid)
        batch_size = config["batch_size"]
        train_mb_itr = util.generate_minibatches((x_train, y_train), 
                                                 batch_size=batch_size)
        valid_mb_itr = util.generate_minibatches((x_valid, y_valid), 
                                                 batch_size=batch_size)
        train_loss = 0
        train_accy = 0
        for _ in range(num_train_samples//batch_size):
            x_train_mb, y_train_mb = next(train_mb_itr)
            loss = model.forward(x_train_mb, targets=y_train_mb)
            yh_train_mb = model.y
            model.backward()
            train_loss += loss
            train_accy += util.calculateCorrect(yh_train_mb, y_train_mb)
        train_loss /= (num_train_samples//batch_size)
        train_accy /= (num_train_samples//batch_size)

        valid_loss = 0
        valid_accy = 0
        for _ in range(num_valid_samples//batch_size):
            x_valid_mb, y_valid_mb = next(valid_mb_itr)
            loss = model.forward(x_valid_mb, targets=y_valid_mb)
            yh_valid_mb = model.y
            valid_loss += loss
            valid_accy += util.calculateCorrect(yh_valid_mb, y_valid_mb)
        valid_loss /= (num_valid_samples//batch_size)
        valid_accy /= (num_valid_samples//batch_size)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accy)
    util.plots(train_losses, train_accuracies, valid_losses, valid_accuracies, earlyStop=19)

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
    test_accy = 0
    for _ in range(num_test_samples//batch_size):
        x_test_mb, y_test_mb = next(test_mb_itr)
        loss = model.forward(x_test_mb, targets=y_test_mb)
        yh_test_mb = model.y
        test_loss += loss
        test_accy += util.calculateCorrect(yh_test_mb, y_test_mb)
    test_loss /= (num_test_samples//batch_size)
    test_accy /= (num_test_samples//batch_size)
    return test_loss, test_accy


