
import tensorflow as tf

class CNNet():
    def __init__(self, input_shape, actions, scale=255.0):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=input_shape))
        self.model.add(tf.keras.layers.Lambda(lambda input: input / scale))
        self.model.add(tf.keras.layers.Conv2D(32,
                                              kernel_size=8,
                                              strides=4,
                                              padding="valid",
                                              activation="relu"))
        self.model.add(tf.keras.layers.Conv2D(64,
                                              kernel_size=4,
                                              strides=2,
                                              padding="valid",
                                              activation="relu"))
        self.model.add(tf.keras.layers.Conv2D(64,
                                              kernel_size=3,
                                              strides=1,
                                              padding="valid",
                                              activation="relu"))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512,
                                             activation="relu"))
        self.model.add(tf.keras.layers.Dense(actions))
