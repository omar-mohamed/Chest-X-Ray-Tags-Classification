import tensorflow as tf

K = tf.keras.backend


class FocalLoss(object):
    def __init__(self, gamma=2, alpha=0.25):
        self._gamma = gamma
        self._alpha = alpha

    def compute_loss(self, y_true, y_pred):
        cross_entropy_loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        p_t = ((y_true * y_pred) +
               ((1 - y_true) * (1 - y_pred)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = tf.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha +
                                   (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)
        return K.mean(focal_cross_entropy_loss, axis=-1)