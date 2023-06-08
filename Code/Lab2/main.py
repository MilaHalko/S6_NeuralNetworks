import numpy as np

from Lab2.NNmodels import *
from Lab2.output import *

# PARAMETERS
N = 1500
TRAIN_PERCENTAGE = 0.8
EPOCHS = 600
LOSS = 'mean_squared_logarithmic_error'
LEARNING_RATE = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.99,
)

# DATA
inputs = np.random.uniform(10, size=(N, 2))
z = np.array([x ** 2 + x * y for x, y in inputs])
inputs_train = inputs[:int(N * TRAIN_PERCENTAGE)]
z_train = z[:int(N * TRAIN_PERCENTAGE)]
inputs_test = inputs[int(N * TRAIN_PERCENTAGE):]
z_test = z[int(N * TRAIN_PERCENTAGE):]


def train_model(model):
    if model.name == 'Feedforward':
        model.compile(loss=LOSS, optimizer=tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE))
    else:
        model.compile(loss=LOSS, optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE))
    history = model.fit(inputs_train, z_train, epochs=EPOCHS, validation_data=(inputs_test, z_test), verbose=0)
    output(model.name, history)


def main():
    set_txt_name('final_results2')
    test_start('N = 2000 \nTRAIN_PERCENTAGE = 0.9 \nEPOCHS = 700 \n' +
               'LOSS = "mean_squared_logarithmic_error" \nACTIVATION = "relu" \nFUNCTION = x^2 + xy')

    train_model(get_feedforward_model([10]))
    train_model(get_feedforward_model([20]))
    train_model(get_cascade_model([20]))
    train_model(get_cascade_model([10, 10]))
    train_model(get_Elman_model([15]))
    train_model(get_Elman_model([5, 5, 5]))


if __name__ == '__main__':
    main()
