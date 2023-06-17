import tensorflow as tf


def get_model(hidden_layers):
    layers = [tf.keras.layers.Flatten()]
    for layer in hidden_layers:
        layers.append(tf.keras.layers.Dense(layer, activation='relu'))
    layers.append(tf.keras.layers.Dense(10, activation='softmax'))
    return tf.keras.Sequential(layers)


def compile_model(model, learning_rate=0.001):
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy'],
    )
