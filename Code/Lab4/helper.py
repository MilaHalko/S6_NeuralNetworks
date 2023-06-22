import matplotlib.pyplot as plt
import tensorflow as tf

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# def load_cifar10(train_size, test_size, validation_size):
#     (train_i, train_l), (test_i, test_l) = tf.keras.datasets.cifar10.load_data()
#     test_i, test_l = test_i[test_size:], test_l[test_size:]
#     valid_i, valid_l = train_i[:validation_size], train_l[validation_size]
#     train_i, train_l = train_i[train_size], train_l[train_size]
#     return (train_i, train_l), (test_i, test_l), (valid_i, valid_l)


def get_dataset_representation(train_size, test_size, valid_size):
    (train_i, train_l), (test_i, test_l) = tf.keras.datasets.cifar10.load_data()
    test_i, test_l = test_i[test_size:], test_l[test_size:]
    valid_i, valid_l = train_i[:valid_size], train_l[valid_size]
    train_i, train_l = train_i[train_size], train_l[train_size]

    train_ds = tf.data.Dataset.from_tensor_slices((train_i, train_l))
    test_ds = tf.data.Dataset.from_tensor_slices((test_i, test_l))
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_i, valid_l))

    return train_ds, test_ds, valid_ds


def print_5images(dataset, count):
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(dataset.take(count)):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(image)
        plt.title(CLASS_NAMES[label.numpy()[0]])
        plt.axis('off')

