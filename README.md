<h1 align="center">
  Multi-layer Neural Networks to classify CIFAR-100 Dataset
</h1>

<p align="center">
  <a href="#Introduction">Introduction</a> •
  <a href="#Usage">Usage</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Introduction

The project aims at implementing simple deep learning models with 1-2 hidden layers for classifying CIFAR-100 dataset. The project is divided into util.py, nueralnet.py, train.py, gradient.py, constants.py and main.py. Data loading, preprocessing is implemented in the util.py file. The network functions, loss function and activation functions are implemented in nueralnet.py file. The train.py file implements the training loop. The main.py integrates all the code at one place.

## Usage

```bash
# Download the data
$ bash get_cifar100data.sh

# Clone this repository
$ https://github.com/soumya123ganguly/PA2

# Go into the repository
$ cd PA2

# Install dependencies
$ pip3 install numpy
$ pip3 install matplotlib
$ pip3 install scikit-learn
$ pip3 install pyyaml
$ pip3 install pandas

# Run the app
$ python main.py --experiment <test_experiment>
```
Change the configs in configs folder for updating the experiment hyperparameters

## Credits

Soumya and Nishanth have collaborated in implemting the project from ground up.

## License

MIT

