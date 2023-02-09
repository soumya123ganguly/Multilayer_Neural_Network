################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2023
# Code by Chaitanya Animesh
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import gradient
from constants import *
from train import *
from gradient import *
import argparse
from neuralnet import *

#TODO
def main(args):

    # Read the required config
    # Create different config files for different experiments
    configFile=None #Will contain the name of the config file to be loaded
    if (args.experiment == 'test_gradients'):  #3b
        configFile = "config_3b.yaml"
    elif(args.experiment=='test_momentum'):  #3c
        configFile = "config_3c.yaml"
    elif (args.experiment == 'test_regularization'): #3d
        configFile = "config_3d.yaml"
    elif (args.experiment == 'test_activation'): #3e
        configFile = "config_3e.yaml"
    elif (args.experiment == 'test_hidden_units'):  #3f-i
        configFile = "config_3f1.yaml"
    elif (args.experiment == 'test_hidden_layers'):  #3f-ii
        configFile = "config_3f2.yaml"
    elif (args.experiment == 'test_100_classes'):  #3g
        configFile = "config_3g.yaml"

    # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = util.load_config(configYamlPath + configFile) # Set configYamlPath, configFile  in constants.py

    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(path=datasetDir, num_classes=config['layer_specs'][-1])  # Set datasetDir in constants.py

    if(args.experiment == 'test_gradients'):
        gradient.checkGradient(x_train, y_train, config)
        return 1

    # Create a Neural Network object which will be our model
    model = Neuralnetwork(config=config)

    # train the model. Use train.py's train method for this
    model = train(model, x_train, y_train, x_valid, y_valid, config)

    # test the model. Use train.py's modelTest method for this
    test_acc, test_loss =  modelTest(model, x_test, y_test)

    # Print test accuracy and test loss
    print('Test Accuracy:', test_acc, ' Test Loss:', test_loss)


if __name__ == "__main__":

    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum', help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)