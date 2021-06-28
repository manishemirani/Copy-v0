import os
import gym
import numpy as np
import tensorflow as tf
import tqdm
import statistics
import collections
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.losses import CategoricalCrossentropy, Huber
from typing import Tuple

classes = ["A", "B", "C", "D", "E", " "]
env = gym.make("Copy-v0")  # making environment
env.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


#  define a function for encoding observations
def encoder(index_number):
    encode_vector = np.zeros(shape=(1, len(classes)))  # create a zero array with shape (1, 6)
    encode_vector[0, index_number] = 1

    return tf.constant(encode_vector, dtype=tf.float32)  # return encode_vector as a tensor


# define Actor model
class ActorModel(tf.keras.Model):
    def __init__(self, num_hidden_state,
                 num_classes):
        super(ActorModel, self).__init__()

        self.hs = LSTM(num_hidden_state, return_sequences=True)  # define lstm
        self.flat = Flatten()  # flatten
        self.movement = Dense(2)  # right or left
        self.write_output = Dense(2)  # true or false
        self.classification = Dense(num_classes)
        self.critic = Dense(1)

    def call(self, inp) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x = inp
        x = x[tf.newaxis, ...]  # E.g x --> (1, x)
        x = self.hs(x)  # pass x to LSTM cells
        x = self.flat(x)  # flatting x
        return self.movement(x), self.write_output(x), self.classification(x), self.critic(x)


# define a function for run episode
def episode(model: tf.keras.Model,
            max_step) -> Tuple[list, list,
                               list, list, list, list]:
    # define memories
    movement_memory = []
    classification_memory = []
    write_output_memory = []
    observation_labels = []
    rewards = []
    values = []
    # define initial observation
    observation = env.reset()
    for i in range(max_step):  # run until the done is True

        observation = encoder(observation)
        observation_labels.append(observation[:, :-1])
        action_logits = model(observation)  # predict probs

        movement_logits, write_output_logits, classification_logits, value = action_logits  # split logits

        values.append(value)  # add value to memory

        movement = tf.random.categorical(movement_logits, 1)[0, 0].numpy()  # chooses random action
        movement_probs = tf.nn.softmax(movement_logits)  # apply softmax function on movement logits
        movement_memory.append(movement_probs[0, movement])  # add movement to memory

        write_output = tf.random.categorical(write_output_logits, 1)[0, 0].numpy()  # choose random write_ouput
        write_output_probs = tf.nn.softmax(write_output_logits)  # # apply softmax function on logits
        write_output_memory.append(write_output_probs[0, write_output])  # add  write_output to memory

        classification = tf.random.categorical(classification_logits, 1)[0, 0].numpy()  # choose random class
        classification_probs = tf.nn.softmax(classification_logits)  # apply softmax function on classes logits

        classification_memory.append(classification_probs)  # add class prob to memory

        action = (movement, write_output, classification)  # combines logits

        observation, reward, done, _ = env.step(action)

        rewards.append(reward)  # add reward to memory
        if tf.cast(done, tf.bool):
            break

    return movement_memory, write_output_memory, classification_memory, rewards, observation_labels, values


# define a function for normalizing discounted rewards
eps = np.finfo(np.float32).eps.item()


def normalize(inputs):
    inputs -= tf.reduce_mean(inputs)
    inputs /= (tf.math.reduce_std(inputs) + eps)  # add eps to preventing dividing by zero

    return tf.cast(inputs, dtype=tf.float32)


# define a function for discounted reward
def discounted_reward(rewards, gamma):
    dis_rewards = np.zeros_like(rewards)  # create an zero array with the same shape like rewards

    discount = 0

    for i in reversed(range(0, len(rewards))):
        discount = rewards[i] + gamma * discount

        dis_rewards[i] = discount

    return normalize(dis_rewards)  # normalizing result


categorical_crossentropy = CategoricalCrossentropy(from_logits=True)
huber_loss = Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(action_probs, observation_labels, rewards):
    movement_probs, write_output_probs, classification_probs, values = action_probs  # split probs
    advantage = rewards - values
    log_movement = tf.math.log(movement_probs)  # apply log on movement probs

    log_write_output = tf.math.log(write_output_probs)  # apply log on write_output probs

    write_output_loss = -tf.math.reduce_sum(log_write_output * advantage)  # compute write_output loss

    movement_loss = -tf.math.reduce_sum(log_movement * advantage)  # compute movement loss

    classification = categorical_crossentropy(observation_labels, classification_probs)  # compute classes loss

    total_loss = write_output_loss + movement_loss + classification  # add losses

    critic_loss = huber_loss(values, rewards)

    return total_loss + critic_loss


def train_step(model: tf.keras.Model, optimizer: Optimizer,
               gamma, max_step_per_episode):
    with tf.GradientTape() as tape:
        movement_probs, write_output_probs, \
        classification_probs, rewards, observation_labels, values = episode(model, max_step_per_episode)

        action_probs = (movement_probs, write_output_probs, classification_probs, values)

        loss = compute_loss(action_probs, observation_labels, discounted_reward(rewards, gamma))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    total_reward = tf.math.reduce_sum(rewards)

    return total_reward


num_classes = len(classes) - 1
num_hidden_state = 100
model = ActorModel(num_hidden_state, num_classes)
optimizer = Adam(learning_rate=0.01)
max_step_per_episode = 1000 
max_episodes = 3500

checkpoint_dir = "./train_checks"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpts")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

reward = 0


def train(model, optimizer, max_step_per_episode,
          max_episodes, min_consecutive_episode=100, reward_threshold=25.72, gamma=0.5):
    episodes_reward: collections.deque = collections.deque(maxlen=min_consecutive_episode)
    with tqdm.trange(max_episodes) as ttr:
        for i in ttr:
            episode_reward = int(train_step(model, optimizer, gamma,
                                            max_step_per_episode))
            if i % 500 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            episodes_reward.append(episode_reward)
            running_reward = statistics.mean(episodes_reward)

            ttr.set_description(f'Episode {i}')
            ttr.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            if running_reward >= reward_threshold and i >= min_consecutive_episode:
                break
    print(f'[INFO] Solved at episode {i}: average reward: {running_reward:.2f}!')


train(model=model, optimizer=optimizer,
      max_step_per_episode=max_step_per_episode,
      max_episodes=max_episodes)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

num_sample = 30
observation = env.reset()
for i in range(30):
    env.render()
    observation = encoder(observation)
    movement, write_output, word, _ = model(observation)

    movement = np.argmax(np.squeeze(movement))
    write_output = np.argmax(np.squeeze(write_output))
    word = np.argmax(np.squeeze(word))

    action = (movement, write_output, word)

    observation, _, done, _ = env.step(action)

    if done:
        observation = env.reset()
