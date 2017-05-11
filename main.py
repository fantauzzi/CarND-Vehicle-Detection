import os
import cv2
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

''' Conveniently collect some program parameters in one place'''


class Params:
    image_width = 64  # Desired width in pixels of images before feature extraction
    image_height = 64  # Desired height in pixels of images before feature extraction
    image_channels = 3  # Desired number of channels in images before feature extraction
    pickled_dataset = 'car-no-car.p'  # File name for the pickled dataset
    dataset_base_dir = '/home/fanta/datasets/vehicles-detection'  # Base dir for the originals (not yet pickled) datasets
    random_seed = 42  # Answer to the Ultimate Question of Life, the Universe, and Everything
    car_label = 1  # Label for a car image
    non_car_label = 0  # Label for a non-car image


def format_image(image):
    """
    Resize the given image as necessary and returns the result. Function used to ensure all images in the dataset
    have the same size before feature extraction.
    """
    height, width, n_channels = image.shape
    assert height == Params.image_width and width == Params.image_height and n_channels == Params.image_channels
    return image


def load_and_pickled_dataset():
    """
    Loads the datasets, converts their images to the desired size and format, assembles them in one big
    dataset and saves it in a pickled file before returning it.
    :return: (x, y) where x is a list of images, and y is the corresponding list of labels. 
    """
    subdirs = ['vehicles/GTI_Far',
               'vehicles/GTI_Left',
               'vehicles/GTI_MiddleClose',
               'vehicles/GTI_Right',
               'non-vehicles/Extras',
               'non-vehicles/GTI']

    ''' 1 if the corresponding element in `subdirs` is a directory with car images, 0 if it is a directory with non-car
    images '''
    subdirs_y = [1, 1, 1, 1, 0, 0]

    dataset_x, dataset_y = [], []
    for subdir, y in zip(subdirs, subdirs_y):
        path_to_subdir = Params.dataset_base_dir + '/' + subdir
        for fname in os.listdir(path_to_subdir):
            if not fname.endswith('.png'):
                continue
            image = cv2.imread(path_to_subdir + '/' + fname)
            assert image is not None
            image = format_image(image)
            dataset_x.append(image)
            dataset_y.append(Params.car_label if y == 1 else Params.non_car_label)

    pickle.dump((dataset_x, dataset_y), open(Params.pickled_dataset, "wb"))
    return dataset_x, dataset_y


def shuffle_and_split_dataset(dataset_x, dataset_y, training_size=.8):
    """
    Splits randomly the given dataset between a training, validation and test sets. 
    :param dataset_x: the dataset data.
    :param dataset_y: the dataset labels.
    :param training_size: the desired training set size, as a fraction of the given dataset size; remaining data
    are split half and half between validation and test set.
    :return: (train_x, train_y, valid_x, valid_y, test_x, test_y)
    """

    assert 0 < training_size < 1
    train_x, test_x, train_y, test_y = train_test_split(dataset_x,
                                                        dataset_y,
                                                        random_state=Params.random_seed,
                                                        test_size=1 - training_size)
    valid_size = len(test_y) // 2
    valid_x = test_x[0:valid_size]
    valid_y = test_y[0:valid_size]
    test_x = test_x[valid_size:]
    test_y = test_y[valid_size:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


if __name__ == '__main__':

    # Read the dataset (and save it pickled, if not done already)
    if not os.path.isfile(Params.pickled_dataset):
        dataset_x, dataset_y = load_and_pickled_dataset()
    else:
        with open(Params.pickled_dataset, mode='rb') as pickle_file:
            dataset_x, dataset_y = pickle.load(pickle_file)

    # Print dataset stats
    n_cars = sum(label == Params.car_label for label in dataset_y)
    print('Read', len(dataset_y), 'images,', n_cars, 'with car, and', len(dataset_y) - n_cars, 'with no car')

    # Split the dataset between training, validation and test sets
    train_x, train_y, valid_x, valid_y, test_x, test_y = shuffle_and_split_dataset(dataset_x, dataset_y)
    print('Training dataset size', len(train_y), ', validation', len(valid_y), ', test', len(test_y))

'''
* Load the dataset(s)
* Crop and resize every image as necessary
* Shuffle the dataset and partition it into training, test and validation data sets
* Save the datasets as a pickle

* Load the datasets from pickle
Augment the training dataset (optional)
* Print dataset stats
Convert every image into a vector of features
Train the SVM
Test the trained SVM on the test data set
Test the trained SVM on the validation data set (optional)
'''
