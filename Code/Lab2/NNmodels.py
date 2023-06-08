import tensorflow as tf

INPUT_SIZE = 2
ACTIVATION = 'relu'


def get_feedforward_model(neurons_in_hidden, input_size=INPUT_SIZE, activation=ACTIVATION):
    layers = inputs = tf.keras.layers.Input(input_size)
    for neurons in neurons_in_hidden:
        layers = tf.keras.layers.Dense(neurons, activation=activation)(layers)
    output = tf.keras.layers.Dense(1)(layers)
    return tf.keras.Model(inputs, output, name="Feedforward")


def get_cascade_model(neurons_in_hidden, input_size=INPUT_SIZE, activation=ACTIVATION):
    layers = inputs = tf.keras.layers.Input(input_size)
    for neurons in neurons_in_hidden:
        layer = tf.keras.layers.Dense(neurons, activation=activation)(layers)
        layers = tf.keras.layers.concatenate([layers, layer])
    output = tf.keras.layers.Dense(1)(layers)
    return tf.keras.Model(inputs, output, name="Cascade")


def get_Elman_model(neurons_in_hidden, input_size=INPUT_SIZE, activation=ACTIVATION):
    inputs = tf.keras.layers.Input(input_size)
    layers = tf.expand_dims(inputs, axis=1)
    layers = tf.keras.layers.SimpleRNN(neurons_in_hidden[0])(layers)
    for neurons in neurons_in_hidden[1:]:
        layers = tf.expand_dims(layers, axis=1)
        layers = tf.keras.layers.SimpleRNN(neurons, activation=activation)(layers)
    output = tf.keras.layers.Dense(1)(layers)
    return tf.keras.Model(inputs, output, name="Elman")
