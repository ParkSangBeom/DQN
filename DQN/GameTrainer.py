import numpy as np
import tensorflow as tf
import random
import gym
import DQN
import os
from collections import deque
from typing import List

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#env = gym.make('CartPole-v1')
env = gym.make('MountainCar-v0')

INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n
DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
MAX_EPISODE = 10000
BATCH_SIZE = 50

TARGET_UPDATE_FREQUENCY = 5

def train_minibatch(mainDQN: DQN.DQN, targetDQN: DQN.DQN, train_batch: list) -> float:
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    return mainDQN.update(X, y)

def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    with tf.Session() as sess:
        mainDQN = DQN.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = DQN.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")

        sess.run(tf.global_variables_initializer())

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        maxCount = 0

        for episode in range(MAX_EPISODE):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()
            
            while not done:

                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)

                if done:
                    if step_count < 10000 - 1:
                        maxCount = 0;

                    reward = -1

                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1

                if step_count >= 10000:
                    maxCount += 1
                    break;
   

            print("[Episode : ", episode, "] [Steps : ", step_count, "]")

            if len(replay_buffer) >= BATCH_SIZE:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    train_minibatch(mainDQN, targetDQN, minibatch)

                sess.run(copy_ops)
            #if step_count % TARGET_UPDATE_FREQUENCY == 0:
            #        sess.run(copy_ops)
            if step_count >= 10000 and maxCount >= 9:
                break;

        while True:
            observation = env.reset()
            reward_sum = 0

            while True:
                env.render()

                a = np.argmax(mainDQN.predict(observation))
                observation, reward, done, _ = env.step(a)
                reward_sum += 1
                if done:
                    print("Reward : ", reward_sum)
                    break;

if __name__ == "__main__":
    main()