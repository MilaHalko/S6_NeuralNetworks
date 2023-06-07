import matplotlib.pyplot as plt
import numpy as np

from Lab2.NNmodels import *

TRAIN_PERCENTAGE = 0.75
EPOCHS = 500
LOSS = 'log_cosh'
LEARNING_RATE = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.99,
)
METRICS = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

inputs = np.random.uniform(0, 10, (1500, 2))
z = np.array([x ** 2 + y ** 2 for x, y in inputs])
inputs_train = inputs[:int(len(inputs) * TRAIN_PERCENTAGE)]
z_train = z[:int(len(z) * TRAIN_PERCENTAGE)]
inputs_test = inputs[int(len(inputs) * TRAIN_PERCENTAGE):]
z_test = z[int(len(z) * TRAIN_PERCENTAGE):]


def train_model(model):
    model.compile(loss=LOSS, optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE), metrics=METRICS)
    model.summary()
    history = model.fit(inputs_train, z_train, epochs=EPOCHS, validation_data=(inputs_test, z_test), verbose=0)
    output(history)


def output(history):
    print("Train Loss:", history.history['loss'][0])
    print("Test Loss:", history.history['val_loss'][0])
    print("Final Loss:", history.history['loss'][-1])
    print("Final MSE:", history.history['mean_squared_error'][-1])
    print("Final MAE:", history.history['mean_absolute_error'][-1])
    print("\n\n\n")

    plt.title('Training loss (log_cosh)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Test loss')
    plt.grid()
    plt.legend()
    plt.show()


def main():
    train_model(get_feedforward_model([10]))
    train_model(get_feedforward_model([20]))
    train_model(get_cascade_model([20]))
    train_model(get_cascade_model([10, 10]))
    train_model(get_Elman_model([15]))
    train_model(get_Elman_model([5, 5, 5]))


if __name__ == '__main__':
    main()
