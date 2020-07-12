
from Network.CNN import CNNet
import tensorflow as tf
import math, random
import numpy as np
import os
import timeit

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Agent():
    def __init__(self, parser):
        self.buffer_size = parser.buffer_size
        self.mode = parser.mode
        self.batch_size = parser.batch_size
        self.learning_rate = parser.learning_rate
        self.algorithm = parser.algorithm
        self.max_epsilon = parser.max_epsilon
        self.min_epsilon = parser.min_epsilon
        self.decay_rate = parser.decay_rate
        self.discount_factor = parser.discount_factor
        self.epsilon = lambda step: \
            (self.max_epsilon - self.min_epsilon) * math.exp(- step / self.decay_rate) + self.min_epsilon

    def initilize(self, input_shape, model_dir, actions, load_model=None, loss="mse", optimizer="Adam"):

        self.model_dir = os.path.join(model_dir, "mymodel")
        self.actions = actions

        # last_10_average_reward = lambda y_pre, y_target: \
        #     sum(self.total_rewards[-10:]) / max(len(self.total_rewards[-10:]), 1)

        if self.mode == "Train":
            self.loss = tf.keras.losses.get(loss)
            if optimizer == "Adam":
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            self.online = CNNet(input_shape, actions)
            self.target = CNNet(input_shape, actions)
            self.update_networks()

    def step(self, state, step):
        if self.mode == "Train":
            if random.random() > self.epsilon(step):
                return self._predict(tf.constant(np.expand_dims(state, axis=0))).numpy()[0]
            else:
                return random.randrange(self.actions)

    # TODO: Add more learning algorithms
    def train(self, replay_buffer):
        if self.mode == "Train":
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
            option = {"DDQN": self._DDQN}
            return option[self.algorithm](tf.constant(state),
                                          tf.constant(action),
                                          tf.constant(reward),
                                          tf.constant(next_state),
                                          tf.constant(done)).numpy()

    @tf.function
    def _DDQN(self, state, action, reward, next_state, done):
        next_Q_values = self.online.model(next_state)
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.actions)
        best_next_Q_values = tf.reduce_sum(next_mask * self.target.model(next_state), axis=1)
        target_Q_values = reward + (1 - done) * best_next_Q_values * self.discount_factor
        mask = tf.one_hot(action, self.actions)
        with tf.GradientTape() as tape:
            all_Q_values = self.online.model(state)
            Q_values = tf.reduce_sum(mask * all_Q_values, axis=1)
            loss = tf.reduce_mean(self.loss(target_Q_values, Q_values))
        gradient = tape.gradient(loss, self.online.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.online.model.trainable_variables))
        return loss

    @tf.function
    def _predict(self, state):
        return tf.argmax(self.online.model(state), axis=1)

    def update_networks(self):
        self.target.model.set_weights(self.online.model.get_weights())

    def save_model(self):
        tf.keras.models.save_model(self.target.model, self.model_dir)


