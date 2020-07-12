
import sys
import os
import json
import numpy as np
import argparse
import gym
# from Wrapper.gym_wrappers import MainGymWrapper
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from pong_trainer.model import Agent
from pong_trainer.util import ReplayBuffer, hp_directory
import tensorflow as tf
import yaml
import datetime
import hyperopt
import pickle
from functools import partial
import math
import shutil

def rl_learner(*args, **kwargs):
    for t in args:
        for key, value in t.items():
            kwargs[key] = value

    print("TUNING HYPERPARAMETERS:")
    print(args)

    parser = argparse.Namespace(**kwargs)
    ratio = 0.5
    weights = np.array([ratio ** 3, ratio ** 2, ratio, 1.0], dtype=np.float32)

    # env = MainGymWrapper.wrap(gym.make(env_name))
    env = make_atari(parser.environment)
    env = wrap_deepmind(env, frame_stack=True)
    env = wrap_pytorch(env)

    if parser.mode == "Train":
        print("STARTING...")
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model_path = "model-" + now
        dir_path = os.path.join(parser.job_dir, model_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print("MODEL WILL BE STORED AT: ", dir_path)

        writer = tf.summary.create_file_writer(dir_path)

        replay_buffer = ReplayBuffer(parser.buffer_size)
        agent = Agent(parser)
        input_shape = env.observation_space.shape
        input_shape = (input_shape[1], input_shape[2], 1)
        agent.initilize(input_shape, dir_path, env.action_space.n)

        all_returns = []
        episode_return = 0
        episode_num = 1
        loss = 0
        state = env.reset()
        state = np.expand_dims(np.average(np.float32(state), axis=0, weights=weights), axis=0)
        state = np.transpose(state, (1, 2, 0))

        for step in range(1, parser.steps+1):
            action = agent.step(state, step)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(np.average(np.float32(next_state), axis=0, weights=weights), axis=0)
            next_state = np.transpose(next_state, (1, 2, 0))
            episode_return += reward

            replay_buffer.push((state, action, reward, next_state, done))

            state = next_state

            if step >= parser.start_train:
                loss = agent.train(replay_buffer)

            if step >= parser.start_train and step % parser.update_target == 0:
                agent.update_networks()
                agent.save_model()

            if step >= parser.start_train and step % parser.log_frequency == 0:
                with writer.as_default():
                    tf.summary.scalar("last_10_average_returns", sum(all_returns[-10:]) / float(max(len(all_returns[-10:]), 1)), step=step)
                    tf.summary.scalar("loss", loss, step=step)
                writer.flush()

            if done:
                print('CURRENT STEP: {}, EPISODE_NUMBER: {}, EPISODE REWARD: {}. EPISODE DONE!'
                      .format(step, episode_num, episode_return))
                all_returns.append(episode_return)
                episode_return = 0
                episode_num += 1
                state = env.reset()
                state = np.expand_dims(np.average(np.float32(state), axis=0, weights=weights), axis=0)
                state = np.transpose(state, (1, 2, 0))

        return {"loss": -sum(all_returns[-10:]) / float(max(len(all_returns[-10:]), 1)),
                "model_dir": dir_path,
                "status": hyperopt.STATUS_OK,
                "attachment": {
                    "return": pickle.dumps(all_returns)
                    # os.path.join(dir_path, "returns.txt"):
                    #     pickle.dump(all_returns, open(os.path.join(dir_path, "returns.txt"), "wb"))
                }}


def _cml_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--environment',
        help='Atari Game Environment to be used',
        type=str,
        default='PongNoFrameskip-v4')
    parser.add_argument(
        '--steps',
        help='Number of steps for the agent to play the game',
        type=int,
        default=10000)
    parser.add_argument(
        '--start_train',
        help='Number of steps after which to start training',
        type=int,
        default=10000)
    parser.add_argument(
        '--update_target',
        help='Number of steps after which to update the target network',
        type=int,
        default=10000)
    parser.add_argument(
        '--buffer_size',
        help='Size of the experience buffer',
        type=int,
        default=100000)
    parser.add_argument(
        '--mode',
        help='Whether we are training the agent or playing the game',
        type=str,
        default="Train")
    parser.add_argument(
        '--job_dir',
        help='Directory where to save the given model',
        type=str,
        default='models')
    parser.add_argument(
        '--batch_size',
        help='Batch size for sampling and training model',
        type=int,
        default=32)
    parser.add_argument(
        '--learning_rate',
        help='Learning rate for for agent and target network',
        type=float,
        default=0.00001)
    parser.add_argument(
        '--algorithm',
        help='Type of algorithm to be implemented',
        type=str,
        default="DDQN")
    parser.add_argument('--max-epsilon', help='Upper bound of epsilon', type=float, default=1.0)
    parser.add_argument(
        '--min_epsilon', help='Lower bound of epsilon', type=float, default=0.01)
    parser.add_argument('--decay-rate', help='Decay rate of epsilon', type=float, default=30000)
    parser.add_argument(
        '--discount_factor',
        help='Discount Factor for TD Learning',
        type=float,
        default=0.99)
    parser.add_argument(
        '--load_model', help='Loads the model', type=str, default=None)
    parser.add_argument(
        '--log_frequency', help='How often to log information', type=int, default=10000
    )
    parser.add_argument('--yaml_file', help='Optimization space', type=str, default="../hyperparam.yaml")
    result, _ = parser.parse_known_args()
    return result

if __name__ == '__main__':
    parser = _cml_parse()
    if parser.mode == "Train":

        hyperparam = yaml.load(open(parser.yaml_file))["trainingInput"]["hyperparameters"]
        max_trials = hyperparam["maxTrials"]

        space = {}
        for dict in hyperparam["params"]:
            if dict["scaleType"] == "choice":
                space[dict["parameterName"]] = hyperopt.hp.choice(dict["parameterName"],
                                                                   [int(value) for value in dict["discreteValues"]])
            elif dict["scaleType"] == "loguniform":
                space[dict["parameterName"]] = hyperopt.hp.loguniform(dict["parameterName"],
                                                                      math.log(float(dict["minValue"])),
                                                                      math.log(float(dict["maxValue"])))

            elif dict["scaleType"] == "uniform":
                space[dict["parameterName"]] = hyperopt.hp.uniform(dict["parameterName"],
                                                                      float(dict["minValue"]),
                                                                      float(dict["maxValue"]))

        if not os.path.exists(parser.job_dir):
            os.makedirs(parser.job_dir)
        print("MODELS WILL BE STORED AT: ", parser.job_dir)

        if os.path.exists(os.path.join(parser.job_dir, "my_model.hyperopt")):
            trials = pickle.load(open(os.path.join(parser.job_dir, "my_model.hyperopt"), "rb"))
            print("SAVED TRIALS FOUND!")

        else:
            trials = hyperopt.Trials()

        print("CLEANING UP INTERRUPTED DIRS.")
        model_dirs = [trial["result"]["model_dir"] for trial in trials.trials]

        for f in os.scandir(parser.job_dir):
            if f.is_dir() and f.path not in model_dirs:
                shutil.rmtree(f)

        print("STARTING FROM TRIALS {}.".format(len(trials.trials) + 1))

        obj = partial(rl_learner, **vars(parser))

        for i in range(len(trials.trials) + 1, max_trials + 1):
            print("STARTING TRIAL {}.".format(i))
            best = hyperopt.fmin(obj, space=space, algo=hyperopt.tpe.suggest, max_evals=i, trials=trials)
            pickle.dump(trials, open(os.path.join(parser.job_dir, "my_model.hyperopt"), "wb"))
            print("FINISH TRIAL {}.".format(i))
            print("BEST TRIAL NOW:")
            print(best)
