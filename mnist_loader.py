"""
    mnist_loader

    A library to load the MNIST image and label data of handwritten
    digits (http://yann.lecun.com/exdb/mnist/). The `load_data_wrapper`
    should be called to parse the data as lists of tuples that can be
    used to train and test feed-forward neural networks.

    Author: Zilvinas Verseckas, 2017
"""

from numpy import fromfile, zeros 
from struct import unpack

"""
    Wraps and returns the MNIST training and testing data as a tuple:
    (`train_data`, `test_data`). Both `train_data` and `test_data`
    are lists of tuples as well. Each of which has the following form:
    (`image_vector`, `label_vector`). The `image_vector` consists of
    28 x 28 = 784 bytes and the `label_vector` is a 10 dimensional
    vector of ints e.g. 5 = (0, 0, 0, 0, 0, 1, 0, 0, 0, 0).
    `train_n` and `tests_n` represent the number of images to read.
"""
def load_data_wrapper(train_n = 60000, tests_n = 10000):
    return (load_data('train/train-images-idx3-ubyte',
                'train/train-labels-idx1-ubyte', train_n),
            load_data('test/t10k-images-idx3-ubyte',
                'test/t10k-labels-idx1-ubyte', tests_n))
"""
    Wraps and returns the custom training data from IDX as a tuple:
    (`train_data`). Both `train_data`is a list of tuples as well.
    The tuples have the following form: (`image_vector`, `label_vector`).
    The `image_vector` consists of 28 x 28 = 784 bytes and the `label_vector`
    is a 10 dimensionalvector of ints e.g. 5 = (0, 0, 0, 0, 0, 1, 0, 0, 0, 0).
    `image_file` is an idx file with images, `label_file` is an idx file with
    labels, `train_n` represents the number of images to read.
"""
def load_train_data_wrapper(image_file, labels_file, train_n = 60000, ):
    return load_data(image_file, labels_file, train_n)

"""
    Loads `n` images and labels from given files `img_fname`,
    `lab_fname` in a described above tuple form.
"""
def load_data(img_fname, lab_fname, n):
    with open(img_fname, 'rb') as imgf, open(lab_fname, 'rb') as labf:
        imgf.seek(16)
        labf.seek(8)
        return read_data(imgf, labf, n)

"""
    Converts a number `lab` to a ten-dimensional vector.
"""
def vector_label(lab):
    lab_vec = zeros(10)
    lab_vec[lab] = 1
    return lab_vec

"""
    Does the reading of data and construction of tuples
"""
def read_data(imgf, labf, n):
    return [(fromfile(imgf, dtype = 'B', count = 784) \
                .reshape(784, 1) / 255,
             vector_label(unpack('B', labf.read(1))[0]) \
                .reshape(10, 1)
            ) for _ in range(n)]
