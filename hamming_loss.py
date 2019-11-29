import tensorflow as tf

K = tf.keras.backend

def hamming_loss(y_true, y_pred):
    diff = tf.cast(y_true - y_pred, dtype=tf.float32)

    # Counting non-zeros in a differentiable way
    epsilon = K.epsilon()
    nonzero = tf.reduce_sum(tf.math.abs(diff / (tf.math.abs(diff) + epsilon)))

    return tf.reduce_mean(nonzero / K.int_shape(y_pred)[-1])