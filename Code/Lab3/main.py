import matplotlib.pyplot as plt
import tensorflow as tf
from model_init import get_model, compile_model

hidden_layers = [50, 50, 50]
num_epochs, numbers = 10, 9
batch_size = 32
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=int(len(x_train) / batch_size),
    decay_rate=(1 / 10) ** (1 / num_epochs)
)

if __name__ == '__main__':
    model = get_model(hidden_layers)
    compile_model(model, learning_rate)
    history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), verbose=0)
    trained_results, right_results, = [y.argmax() for y in model.predict(x_test[:numbers])], y_test[:numbers]

    model.summary()
    print(f'Accuracy: {history.history["accuracy"][-1]}')
    print(f'Loss: {history.history["loss"][-1]}')
    for i in range(numbers):
        print(f'Predicted: {trained_results[i]} | Actual: {right_results[i]}')
        plt.figure(figsize=(1, 1))
        plt.imshow(x_test[i])
        plt.show()
