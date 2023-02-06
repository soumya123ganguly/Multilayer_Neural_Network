import numpy as np
from neuralnet import Neuralnetwork

def check_grad(model, x_train, y_train):

    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    eps = 1e-2
    l = 1
    n = 1
    model.layers[l].w[-1, n] += eps
    E_wp = model.forward(x_train, targets=y_train)
    model.layers[l].w[-1, n] -= 2*eps
    E_wm = model.forward(x_train, targets=y_train)
    model.layers[l].w[-1, n] += eps
    model.forward(x_train, targets=y_train)
    model.backward(gradReqd=False)
    dw = model.layers[l].dw
    print("&", (E_wp-E_wm)/(2*eps), "\t&", dw[-1, n], "\t&", dw[-1, n]-((E_wp-E_wm)/(2*eps)))

def checkGradient(x_train,y_train,config):

    subsetSize = 1  #Feel free to change this
    sample_idx = np.random.randint(0, len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)