import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from helper import *

TRAIN_SIZE = 5000
TEST_SIZE = 5000
VALID_SIZE = 5000

train_ds, test_ds, valid_ds = get_dataset_representation(TRAIN_SIZE, TEST_SIZE, VALID_SIZE)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

