import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import os
import tensorflow.python.framework.ops as ops


# Constants and matmul
def f1():
    print("Constants and matmul")
    a = tf.constant([[10, 10], [11., 1.]])
    x = tf.constant([[1., 0.], [0., 1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print(y.numpy())


# Tensors, shape and dtype
def f2():
    print("Tensors, shape and dtype")
    zeros_tensor = tf.zeros([3, 3])
    print(zeros_tensor)
    print(zeros_tensor.shape)
    print(zeros_tensor.dtype)


# Variables and assign
def f3():
    print("Variables and assign")
    zeros_tensor = tf.zeros([3, 3])
    v = tf.Variable(zeros_tensor)
    v.assign(zeros_tensor)  # updates value of `v`
    print(v.numpy())


# Base functions
def f4():
    print("Base functions")
    x = tf.constant(10.0, dtype=tf.float32)
    f = 1 + 2 * x + tf.pow(x, 2)
    result = f.numpy()
    print(result)


# Sigmoid function
def f5():
    print("Calculating sigmoid")
    x = tf.constant(np.linspace(-5, 5), dtype=tf.float32)
    sigma = 1 / (1 + tf.exp(-x))
    print(sigma.numpy())


# Linear regression & Machine learning
def f6():
    print("Linear regression & Machine learning")
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + np.random.normal(size=len(x))
    plt.plot(x, y)
    plt.show()

    train_idxes = np.random.choice(range(len(x)), 3 * len(x) // 4)
    test_idxes = np.delete(np.arange(len(x)), train_idxes)
    X_train, Y_train = x[train_idxes], y[train_idxes]
    X_test, Y_test = x[test_idxes], y[test_idxes]
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(len(X_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(len(X_test))

    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_errors = []
    test_errors = []

    for _ in tqdm.tqdm(range(100)):
        train_loss.reset_states()
        for inputs, labels in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

        test_loss.reset_states()
        for inputs, labels in test_dataset:
            predictions = model(inputs)
            loss = loss_object(labels, predictions)
            test_loss(loss)

        train_errors.append(train_loss.result())
        test_errors.append(test_loss.result())

    plt.plot(list(range(100)), train_errors, label='Train')
    plt.plot(list(range(100)), test_errors, label='Test')
    plt.legend()
    plt.savefig('Lin_reg.png')
    plt.show()

    model_output = model.predict(x.reshape((len(x), 1)))
    plt.plot(x, y)
    plt.plot(x, model_output)
    plt.savefig('Lr_forward_pass.png')
    plt.show()


# Linear regression & Machine learning disable_eager_execution
def f7():
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.InteractiveSession()
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + np.random.normal(size=len(x))

    train_idxes = np.random.choice(range(len(x)), 3 * len(x) // 4)
    test_idxes = np.delete(np.arange(len(x)), train_idxes)
    X_Train, Y_Train = x[train_idxes], y[train_idxes]
    X_Test, Y_Test = x[test_idxes], y[test_idxes]
    x_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="input")
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="output")

    model_output = tf.Variable(tf.random.normal([1]), name='bias') + tf.Variable(tf.random.normal([1]), name='k') * x_
    loss = tf.reduce_mean(tf.pow(y_ - model_output, 2))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.0001)
    train_step = optimizer.minimize(loss)

    sess.run(tf.compat.v1.global_variables_initializer())

    n_epochs = 100
    train_errors = []
    test_errors = []

    for i in tqdm.tqdm(range(n_epochs)):
        _, train_err = sess.run([train_step, loss], feed_dict={x_: X_Train.reshape((len(X_Train), 1)),
                                                               y_: Y_Train.reshape((len(Y_Train), 1))})
        train_errors.append(train_err)
        test_err = sess.run(loss,
                            feed_dict={x_: X_Test.reshape((len(X_Test), 1)), y_: Y_Test.reshape((len(Y_Test), 1))})
        test_errors.append(test_err)

    plt.plot(x, y)
    plt.show()

    plt.plot(list(range(n_epochs)), train_errors, label='Train')
    plt.plot(list(range(n_epochs)), test_errors, label='Test')
    plt.legend()
    plt.savefig('lin_reg.png')
    plt.show()

    plt.plot(x, y)
    plt.plot(x, sess.run(model_output, feed_dict={x_: x.reshape((len(x), 1))}))
    plt.savefig("lr_forward_pass.png")
    plt.show()


# Polynomial Regression & Saver & Tensorboard
def f8():
    print("Polynomial Regression & Saver & Tensorboard")
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.InteractiveSession()
    order = 30

    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + np.random.normal(size=len(x))

    ss = StandardScaler()
    poly_features = PolynomialFeatures(degree=order - 1)
    x_poly = ss.fit_transform(poly_features.fit_transform(x.reshape((1000, 1))))

    shuffle_idxes = np.arange(len(x_poly))
    np.random.shuffle(shuffle_idxes)

    X_Train, Y_Train = x_poly[shuffle_idxes[:3 * len(x_poly) // 4]], y[shuffle_idxes[:3 * len(x_poly) // 4]]
    X_Test, Y_Test = x_poly[shuffle_idxes[3 * len(x_poly) // 4:]], y[shuffle_idxes[3 * len(x_poly) // 4:]]

    x_ = tf.compat.v1.placeholder(tf.float32, shape=[None, order], name="input")
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="output")
    w = tf.Variable(tf.random.normal([order, 1]), name='weights')

    model_output = tf.matmul(x_, w)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=starter_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )(global_step)
    loss = tf.reduce_mean(tf.pow(y_ - model_output, 2) + 0.85 * tf.nn.l2_loss(w) + 0.15 * tf.reduce_mean(tf.abs(w)))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    sess.run(tf.compat.v1.global_variables_initializer())

    n_epochs = 1000
    train_errors = []
    test_errors = []

    for i in tqdm.tqdm(range(n_epochs)):
        _, train_err = sess.run([train_step, loss], feed_dict={x_: X_Train, y_: Y_Train.reshape((len(Y_Train), 1))})
        train_errors.append(train_err)
        test_errors.append(sess.run(loss, feed_dict={x_: X_Test, y_: Y_Test.reshape((len(Y_Test), 1))}))

    print("Train error: ", train_errors[:10])
    print("Test error: ", test_errors[:10])

    plt.plot(list(range(n_epochs)), train_errors, label='Train')
    plt.plot(list(range(n_epochs)), test_errors, label='Test')
    plt.legend()
    plt.savefig('poly_reg.png')
    plt.show()

    plt.plot(x, y)
    plt.plot(x, sess.run(model_output, feed_dict={x_: x_poly.reshape((len(x), order))}))
    plt.savefig("pr_forward_pass.png")
    plt.show()

    saver = tf.compat.v1.train.Saver()
    saver.save(sess, "checkpoint_dir/polyModel.ckpt")

    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint_dir/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Model restored: ", ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)


# Neural Network & Layers & ReLU & Bias & Weights
def f9():
    print("Neural Network & Layers & ReLU & Bias & Weights")
    tf.compat.v1.disable_eager_execution()
    ops.reset_default_graph()
    sess = tf.compat.v1.Session()

    x_ = tf.compat.v1.placeholder(name="input", shape=[None, 2], dtype=tf.float32)
    y_ = tf.compat.v1.placeholder(name="output", shape=[None, 1], dtype=tf.float32)

    hidden_neurons = 15
    w1 = tf.Variable(tf.random.uniform(shape=[2, hidden_neurons]))
    b1 = tf.Variable(tf.constant(value=0.0, shape=[hidden_neurons], dtype=tf.float32))
    layer1 = tf.nn.relu(tf.add(tf.matmul(x_, w1), b1))

    w2 = tf.Variable(tf.random.uniform(shape=[hidden_neurons, 1]))
    b2 = tf.Variable(tf.constant(value=0.0, shape=[1], dtype=tf.float32))
    nn_output = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001)
    loss = tf.reduce_mean(tf.square(nn_output - y_))
    train_step = optimizer.minimize(loss)
    sess.run(tf.compat.v1.global_variables_initializer())

    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    for _ in range(20000):
        sess.run(train_step, feed_dict={x_: x, y_: y})

    print(sess.run(nn_output, feed_dict={x_: x}))



