import tensorflow as tf
import numpy as np

def atanh(delta=1):
    def _atanh(x):
        return tf.math.atanh(x)
    return _atanh

x = np.array([-10, 1, 30, 12])
atanh_f = atanh()
delta= 1e-4

with tf.Session() as sess:
    x = tf.cast(x, tf.float32)
    x = tf.nn.softmax(x)
    print(x.eval())
    y = atanh_f(x)
    print(y.eval())