from stable_baselines import PPO2
import gym

import time
from importlib import import_module
import argparse
import numpy as np
import os
import yaml
import shutil
import inspect


class EnvRunner:
    def __init__(self, version, episode=False, folder=False,
                 record_video=False, deterministic=False, free_cam=False,
                 no_sleep=False, gui=True, n_recurrent=0):
        self.free_cam = free_cam
        self.gui = gui
        self.no_sleep = no_sleep
        self.version = version
        self.episode = episode
        self.record_video = record_video
        self.folder = folder
        self.deterministic = deterministic
        self.model_name = "aslaug_{}".format(version)
        self.n_recurrent = n_recurrent
        self.obs_list = []
        self.act_list = []
        # Load model
        self.load_model()

        # Load environment
        self.load_env()

        # Prepare pretty-print
        np.set_printoptions(precision=2, suppress=True, sign=' ')

    def set_param(self, param_str):
        param, val = param_str.split(":")
        self.env.set_param(param, float(val))

    def load_env(self):
        # Load module
        mod_path = "data.saved_models.{}.{}".format(folder, self.model_name)
        # mod_path = "envs.{}".format(self.model_name)
        mod_file_path = "data/saved_models/{}".format(folder)

        base_path = mod_file_path + "/aslaug_base.py"
        if not os.path.exists(base_path):
            print("Deprecated saved model! Copying base...")
            a = shutil.copy2("envs/aslaug_base.py", base_path)
            print(a)

        aslaug2d_mod = import_module(mod_path)

        param_path = mod_file_path + "/params.yaml"
        params = None
        if os.path.exists(param_path):
            with open(param_path) as f:
                params = yaml.load(f)["environment_params"]
        # Load env
        recording = record_video is not False
        if "params" in inspect.getargspec(aslaug2d_mod.AslaugEnv).args:
            env = aslaug2d_mod.AslaugEnv(folder_name=self.folder, gui=self.gui,
                                         free_cam=self.free_cam,
                                         recording=recording,
                                         params=params)
        else:
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
        self.fps_queue = 400 * [0.04]
        self.fps_NN_queue = 1000 * [0.04]
        for episode in range(n_episodes):
            self.act_list = []
            self.obs_list = []
            self.episode_id = episode + 1
            obs = self.reset()
            if self.record_video:
                self.obs_list.append(obs.tolist())
            if self.n_recurrent > 0:
                self.obs_hist = [obs]*self.n_recurrent
            self.done = False
            while not self.done:
                ts = time.time()
                self.step()
                self.render()
                dt = time.time() - ts
                if not self.no_sleep and self.env.p["world"]["tau"] - dt > 0:
                    time.sleep(self.env.p["world"]["tau"] - dt)
                self.fps_queue.pop(0)
                self.fps_queue.append(dt)

            # Save world and setpoint history
            if self.record_video:
                prefix, infix = self.env.file_prefix, self.env.file_infix
                ep_id = self.env.episode_id
                self.env.save_world(self.vid_n, prefix, infix, ep_id)

                fn = "{}.video.{}.video{:06}.obs_acts.yaml".format(prefix,
                                                                   infix,
                                                                   ep_id)
                obs_act_path = os.path.join(self.vid_n, fn)
                data = {"observations": self.obs_list,
                        "actions": self.act_list}
                with open(obs_act_path, 'w') as f:
                    yaml.dump(data, f)

    def reset(self, init_state=None, init_setpoint_state=None,
              init_obstacle_grid=None, init_ol=None):
        self.obs = self.env.reset()
        self.done = False
        self.cum_reward = 0.0
        self.n_sp_tot = 0
        return self.obs

    def step(self, print_status=True):
        ts_NN = time.time()
        if self.n_recurrent > 0:
            self.action, _ = self.model.predict(np.array(self.obs_hist),
                                                deterministic=self.deterministic)
            self.action = self.action[-1, :]
        else:
            self.action, _ = self.model.predict(self.obs,
                                                deterministic=self.deterministic)
        te_NN = time.time()
        self.fps_NN_queue.pop(0)
        self.fps_NN_queue.append(te_NN - ts_NN)
        self.obs, self.reward, self.done, self.info = self.env.step(self.action)

        if self.record_video:
            self.obs_list.append(self.obs.tolist())
            self.act_list.append(self.action.tolist())
        if self.n_recurrent > 0:
            self.obs_hist.pop(0)
            self.obs_hist.append(self.obs)
        self.cum_reward += self.reward
        if print_status:
            obs = self.obs
            if hasattr(self.env, "obs_slicing"):
                sl = self.env.obs_slicing
                print("Obs slicing: {}".format(sl))
                obs = ("Setpoint:\n{}\nMBvel:\n{}\nLinkpos:\n{}\n" +
                       "Jointpos:\n{}\nJointvel:\n{}\nScan_f:\n{}\n" +
                       "Scan_r:\n{}\n"
                       ).format(obs[sl[0]:sl[1]], obs[sl[1]:sl[2]],
                                obs[sl[2]:sl[3]], obs[sl[3]:sl[4]],
                                obs[sl[4]:sl[5]], obs[sl[5]:sl[6]],
                                obs[sl[6]:sl[7]])
                succ_rate = self.env.calculate_success_rate()

            print("===============================\n",
                  "Observations\n{}\n\n".format(obs),
                  "Actions\n{}\n".format(self.action),
                  "Reward: {}\n".format(self.reward),
                  "Cum. reward: {}\n".format(self.cum_reward),
                  "Success rate: {}\n".format(succ_rate),
                  "Episode: {}\n".format(self.episode_id),
                  "Setpoint: {}\n".format(self.env.episode_counter),
                  "FPS total: {}\n".format(len(self.fps_queue)/sum(self.fps_queue)),
                  "FPS NN: {}\n".format(len(self.fps_NN_queue)/sum(self.fps_NN_queue)),
                  "\n\nGoal distance: {}\n".format(self.env.last_eucl_dis),
                  "===============================\n\n\n")

    def render(self):
        self.env.render(w=720, h=480)  # 'human_fast', 600, 400

    def close(self):
        self.env.close()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", help="Define version of env to use.")
parser.add_argument("-f", "--folder", help="Specify folder to use.")
# parser.add_argument("-e", "--episode", help="Specify exact episode to use.")
parser.add_argument("-r", "--record_video", help="Specify recording folder.")
parser.add_argument("-n", "--n_episodes", help="Specify number of episodes.",
                    default="50")
parser.add_argument("-cfr", "--copy_from_remote", help="Specify if files should be downloaded from mapcompute first.")
parser.add_argument("-det", "--deterministic", help="Set deterministic or probabilistic actions.", default="False")
parser.add_argument("-fcam", "--free_cam", help="Set camera free.", default="False")
parser.add_argument("-nosleep", "--no_sleep", help="Set camera free.", default="False")
parser.add_argument("-nogui", "--no_gui", help="Set camera free.", default="False")
parser.add_argument("-p", "--param", action='append',
                    help="Set a specific param a-priori. Example to adjust \
                    parameter p[reward][1] to 12: -p reward.1:12")
args = parser.parse_args()

version = args.version
folder = args.folder
ep = None
if folder is not None and len(folder.split(":")) > 1:
    folder, ep = folder.split(":")
record_video = args.record_video
n_episodes = int(args.n_episodes)
free_cam = True if args.free_cam in ["True", "true", "1"] else False
deterministic = True if args.deterministic in ["True", "true", "1"] else False
no_sleep = True if args.no_sleep in ["True", "true", "1"] else False
no_gui = True if args.no_gui in ["True", "true", "1"] else False
param = args.param
if version is None:
    print("Please specify a version. Example: -v v8")

if args.copy_from_remote in ['1', 'True', 'true']:
    os.system("rsync -rav -e ssh --exclude '*.pkl' mapcompute:~/aslaug3d_simulation/data/saved_models/{} data/saved_models".format(folder))
    os.system("rsync -rav -e ssh mapcompute:~/aslaug3d_simulation/data/saved_models/{}/aslaug_{}_{}.pkl data/saved_models/{}".format(folder, version, ep, folder))

if args.copy_from_remote in ['2']:
    os.system("rsync -rav -e ssh wgserver:~/aslaug3d_simulation/data/saved_models/{} data/saved_models".format(folder))

er = EnvRunner(version, ep, folder, record_video, deterministic, free_cam,
               no_sleep, not no_gui)

# Prepare curriculum learning
if param is not None:
    for param_str in param:
        er.set_param(param_str)


print("=======================================\n",
      "Version: {}\n".format(version),
      "Episode: {}\n".format(ep),
      "Folder: {}\n".format(folder),
      "Deterministic: {}\n".format(deterministic),
      "=======================================\n")
er.run_n_episodes(n_episodes)
er.close()
