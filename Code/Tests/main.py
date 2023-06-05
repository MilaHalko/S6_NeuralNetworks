import tensorflow as tf
def f1():
    print("Constants and matmul")
    a = tf.constant([[10, 10], [11., 1.]])
    x = tf.constant([[1., 0.], [0., 1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print(y.numpy())

def f2():   #12
    print("Tensors, shape and dtype")
    zeros_tensor = tf.zeros([3, 3])
    print(zeros_tensor)
    print(zeros_tensor.shape)
    print(zeros_tensor.dtype)

f2()
