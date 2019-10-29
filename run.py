from stable_baselines import PPO2
import gym

import time
from importlib import import_module
import argparse
import numpy as np
import os


class EnvRunner:
    def __init__(self, version, episode=False, folder=False,
                 record_video=False, deterministic=False, free_cam=False,
                 no_sleep=False, gui=True):
        self.free_cam = free_cam
        self.gui = gui
        self.no_sleep = no_sleep
        self.version = version
        self.episode = episode
        self.record_video = record_video
        self.folder = folder
        self.deterministic = deterministic
        self.model_name = "aslaug_{}".format(version)

        # Load environment
        self.load_env()

        # Load model
        self.load_model()

        # Prepare pretty-print
        np.set_printoptions(precision=2, suppress=True, sign=' ')

    def load_env(self):
        # Load module
        #mod_path = "data.saved_models.{}.{}".format(folder, self.model_name)
        mod_path = "envs.{}".format(self.model_name)
        aslaug2d_mod = import_module(mod_path)

        # Load env
        recording = record_video is not False
        env = aslaug2d_mod.AslaugEnv(folder_name=self.folder, gui=self.gui,
                                     free_cam=self.free_cam,
                                     recording=recording)
        if self.record_video:
            vid_n = "data/recordings/{}/{}".format(self.model_name,
                                                   self.record_video)
            env = gym.wrappers.Monitor(env, vid_n,
                                       video_callable=lambda episode_id: True,
                                       force=True)
            self.vid_n = vid_n
        self.env = env
        self.done = False

    def load_model(self):
        model_path = "data/saved_models/"
        if folder:
            model_path = model_path + self.folder + "/"
        else:
            model_path = model_path + self.model_name + "/"

        model_path = model_path + self.model_name

        if self.episode:
            model_path = model_path + "_" + self.episode + ".pkl"

        self.model = PPO2.load(model_path)

    def run_n_episodes(self, n_episodes=1):
        self.n_success = 0
        for episode in range(n_episodes):
            print("Running episode {}.".format(episode+1))
            self.reset()
            self.done = False
            while not self.done:
                ts = time.time()
                self.step()
                self.render()
                dt = time.time()-ts
                if not self.no_sleep and 0.02 - dt > 0:
                    time.sleep(0.02 - dt)

            # Save world and setpoint history
            if self.record_video:
                self.env.save_world(self.vid_n, self.env.file_prefix,
                                    self.env.file_infix, self.env.episode_id)

    def reset(self, init_state=None, init_setpoint_state=None,
              init_obstacle_grid=None, init_ol=None):
        self.obs = self.env.reset()
        self.done = False
        self.cum_reward = 0.0
        return self.obs

    def step(self, print_status=True):
        self.action, _ = self.model.predict(self.obs,
                                            deterministic=self.deterministic)

        self.obs, self.reward, self.done, self.info = self.env.step(self.action)
        self.cum_reward += self.reward
        if print_status:
            obs = self.obs
            if hasattr(self.env, "obs_slicing"):
                sl = self.env.obs_slicing
                obs = ("Setpoint:\n{}\nMBvel:\n{}\nLinkpos:\n{}\n" +
                       "Jointpos:\n{}\nJointvel:\n{}\nScan:\n{}\n"
                       ).format(obs[sl[0]:sl[1]], obs[sl[1]:sl[2]],
                                obs[sl[2]:sl[3]], obs[sl[3]:sl[4]],
                                obs[sl[4]:sl[5]], obs[sl[5]:sl[6]])
                succ_rate = self.env.calculate_success_rate()
            print("===============================\n",
                  "Observations\n{}\n\n".format(obs),
                  "Actions\n{}\n".format(self.action),
                  "Reward: {}\n".format(self.reward),
                  "Cum. reward: {}\n".format(self.cum_reward),
                  "Success rate: {}\n".format(succ_rate),
                  "===============================\n\n\n")

    def render(self):
        self.env.render(w=720, h=480)  # 'human_fast', 600, 400

    def close(self):
        self.env.close()


class EnvComparer:
    def __init__(self, version, folder_1, folder_2, episode=False,
                 record_video=False, deterministic=False):
        self.version = version
        self.folder_1 = folder_1
        self.folder_2 = folder_2
        self.episode = episode
        self.record_video = record_video
        self.deterministic = deterministic
        self.create_envs()

    def create_envs(self):
        self.env1 = EnvRunner(self.version, episode=self.episode,
                              folder=self.folder_1,
                              record_video=self.record_video,
                              deterministic=self.deterministic)
        self.env2 = EnvRunner(self.version, episode=self.episode,
                              folder=self.folder_2,
                              record_video=self.record_video,
                              deterministic=self.deterministic)

    def run_n_episodes(self, n_episodes=1):
        for episode in range(n_episodes):
            print("Running episode {}.".format(episode+1))
            self.reset()
            self.done = False
            while not self.done:
                ts = time.time()
                self.step()
                if not self.record_video:
                    self.render()
                dt = time.time()-ts
                if 0.02 - dt > 0:
                    time.sleep(0.02 - dt)

    def reset(self):
        self.env1.reset()
        if self.record_video:
            env1 = self.env1.env.env
        else:
            env1 = self.env1.env
        state = env1.state.copy()
        setpoint_state = env1.setpoint_state.copy()
        obstacle_grid = env1.obstacle_grid.copy()
        obstacle_locations = env1.obstacle_locations.copy()
        self.env2.reset(init_state=state, init_setpoint_state=setpoint_state,
                        init_obstacle_grid=obstacle_grid,
                        init_ol=obstacle_locations)
        # env2.state = env1.state.copy()
        # env2.setpoint_state = env1.setpoint_state.copy()
        # env2.obstacle_locations = env1.obstacle_locations.copy()
        # env2.obstacle_grid = env1.obstacle_grid.copy()

        self.done = False

    def step(self):
        if not self.env1.done:
            self.env1.step()
        if not self.env2.done:
            self.env2.step()
        self.done = self.env1.done and self.env2.done

    def render(self):
        self.env1.render()
        self.env2.render()

    def close(self):
        self.env1.close()
        self.env2.close()


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", help="Define version of env to use.")
parser.add_argument("-f", "--folder", help="Specify folder to use.")
parser.add_argument("-e", "--episode", help="Specify exact episode to use.")
parser.add_argument("-r", "--record_video", help="Specify recording folder.")
parser.add_argument("-n", "--n_episodes", help="Specify number of episodes.",
                    default="50")
parser.add_argument("-f2", "--folder_2", help="Specify a second agent.")
parser.add_argument("-cfr", "--copy_from_remote", help="Specify if files should be downloaded from mapcompute first.")
parser.add_argument("-det", "--deterministic", help="Set deterministic or probabilistic actions.", default="False")
parser.add_argument("-fcam", "--free_cam", help="Set camera free.", default="False")
parser.add_argument("-nosleep", "--no_sleep", help="Set camera free.", default="False")
parser.add_argument("-nogui", "--no_gui", help="Set camera free.", default="False")
args = parser.parse_args()

version = args.version
ep = args.episode
folder = args.folder
folder_2 = args.folder_2
record_video = args.record_video
n_episodes = int(args.n_episodes)
free_cam = True if args.free_cam in ["True", "true", "1"] else False
deterministic = True if args.deterministic in ["True", "true", "1"] else False
no_sleep = True if args.no_sleep in ["True", "true", "1"] else False
no_gui = True if args.no_gui in ["True", "true", "1"] else False

if version is None:
    print("Please specify a version. Example: -v v8")

if args.copy_from_remote in ['1', 'True', 'true']:

    os.system("rsync -rav -e ssh mapcompute:~/aslaug3d_simulation/data/saved_models/{} data/saved_models".format(folder))


if folder_2:
    er = EnvComparer(version, folder, folder_2, ep,
                     record_video, deterministic)
else:
    er = EnvRunner(version, ep, folder, record_video, deterministic, free_cam,
                   no_sleep, not no_gui)


print("=======================================\n",
      "Version: {}\n".format(version),
      "Episode: {}\n".format(ep),
      "Folder: {}\n".format(folder),
      "Deterministic: {}\n".format(deterministic),
      "=======================================\n")
er.run_n_episodes(n_episodes)
er.close()
