import tensorflow as tf
from tensorflow.keras import layers
class NoisyDense(layers.Layer):
    def __init__(self, units, std_init=0.4, **kwargs):
        self.units = units
        self.std_init = std_init
        super(NoisyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

        self.w_noise = self.add_weight(shape=(input_shape[-1], self.units),
                                       initializer='random_normal',
                                       trainable=False)
        self.b_noise = self.add_weight(shape=(self.units,),
                                       initializer='random_normal',
                                       trainable=False)
        super(NoisyDense, self).build(input_shape)

    def call(self, inputs, training=None):
        if training:
            w = self.w + self.std_init * tf.sign(self.w_noise) * tf.sqrt(tf.abs(self.w_noise))
            b = self.b + self.std_init * tf.sign(self.b_noise) * tf.sqrt(tf.abs(self.b_noise))
        else:
            w = self.w
            b = self.b
        return tf.matmul(inputs, w) + b
