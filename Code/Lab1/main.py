import numpy as np
import tensorflow as tf

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

xor_model = tf.keras.models.Sequential()
xor_model.add(tf.keras.layers.Dense(5, input_dim=2, activation='relu'))
xor_model.add(tf.keras.layers.Dense(5, activation='relu'))
xor_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
xor_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
xor_model.fit(x, y, epochs=1000, verbose=0)
loss_and_accuracy_scores = xor_model.evaluate(x, y)

print(xor_model.summary())
print("--------------------")
print(loss_and_accuracy_scores)
print("--------------------")
print(xor_model.predict(x))
