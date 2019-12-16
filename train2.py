from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from millify import millify
import numpy as np

from importlib import import_module
import argparse
import os
import shutil
import json
import keras
from threading import Lock
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Reshape, Lambda
from keras.models import Model
# Prepare callback global parameters
n_steps = 0
model_idx = 0
cl_idx = 0
info_idx = 0


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", help="Define # steps for learning.",
                        default=10e6)
    parser.add_argument("-n", "--n_cpu", help="Define # processes to use.",
                        default=32)
    parser.add_argument("-v", "--version", help="Set env version.")
    parser.add_argument("-p", "--policy", help="Define policy to use (path).",
                        default="stable_baselines.common.policies.MlpPolicy")
    parser.add_argument("-cp", "--check_point", help="# steps in between \
                        checkpoints",
                        default=500e3)
    parser.add_argument("-f", "--folder", help="Name the folder",
                        default="None")
    parser.add_argument("-de", "--default_env", help="Use default gym env.",
                        default="None")
    parser.add_argument("-cl", "--curriculum_learning", action='append',
                        help="Enable curriculum learning. Example to adjust \
                        parameter reward.r1 from 1 to 5 in 3M steps: -cl reward.r1:1:5:3e6.")
    parser.add_argument("-pt", "--proceed_training",
                        help="Specify model from which training shall be \
                        proceeded. Format: folder_name:episode")
    args = parser.parse_args()

    n_cpu, version = int(float(args.n_cpu)), args.version
    steps = int(float(args.steps))
    policy_arg = args.policy
    n_cp = args.check_point
    folder_name = args.folder
    cl = args.curriculum_learning
    pt = args.proceed_training

    # Prepare module import
    model_name = "aslaug_{}".format(version)
    policy_mod_name = ".".join(policy_arg.split(".")[:-1])
    policy_name = policy_arg.split(".")[-1]

    # Load module for policy
    policy_mod = import_module(policy_mod_name)
    policy = getattr(policy_mod, policy_name)

    # Prepare directory and rename terminal
    if folder_name == "None":
        folder_name = model_name

    os.system("tmux rename-session {}-{}".format(version, folder_name))

    dir_path = "data/saved_models/{}/".format(folder_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    elif len(os.listdir(dir_path)) == 0:
        print("Directory exists, but is empty. Proceeding.")
    else:
        print("Attention, {} already exists.".format(folder_name))
        resp = input("Move it [m], delete it [r] or cancel [c]: ")
        if resp == 'c':
            exit()
        elif resp == 'r':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        elif resp == 'm':
            resp = input("Enter new folder name for {}: ".format(folder_name))
            shutil.move(dir_path, "data/saved_models/{}/".format(resp))
            os.mkdir(dir_path)
        else:
            print("Can't understand your expression.")
            exit()

    # Prepare curriculum learning
    if cl is not None:
        cl_list = []
        for clstr in cl:
            cl_param, cl_start, cl_end, cl_steps = clstr.split(":")
            cl_list.append({"param": cl_param, "start": float(cl_start),
                            "end": float(cl_end), "steps": float(cl_steps)})

    # Prepare custom learning rate function
    def create_custom_lr(lr_max, lr_min, a, b):
        m = (lr_max - lr_min) / (a - b)
        c = lr_max - m * a

        return lambda x: np.min([lr_max, np.max([lr_min, m * x + c])])

    # Prepare model and env
    aslaug_mod = import_module("envs." + model_name)
    shutil.copy("envs/{}.py".format(model_name), dir_path + model_name + ".py")

    with open("params.json") as f:
        params_all = json.load(f)
    learning_params = params_all["learning_params"]
    env_params = params_all["environment_params"]

    print("Learning params: {}".format(learning_params))
    print("Env params: {}".format(env_params))

    # Save learning params to file
    params_file = "data/saved_models/{}/params.json".format(folder_name)
    shutil.copy("params.json", params_file)

    # Save curriculum learning to file
    if cl is not None:
        cl_file = "data/saved_models/{}/curriculum_learning.json".format(folder_name)
        with open(cl_file, 'w') as outfile:
            json.dump(cl_list, outfile)

    # Copy policy to models folder
    shutil.copy(policy_mod.__file__,
                "data/saved_models/{}/{}.py".format(folder_name, policy_name))

    if type(learning_params["learning_rate"]) in [list, tuple]:
        lr_params = learning_params["learning_rate"]
        learning_params["learning_rate"] = create_custom_lr(*lr_params)
    if type(learning_params["cliprange"]) in [list, tuple]:
        lr_params = learning_params["cliprange"]
        learning_params["cliprange"] = create_custom_lr(*lr_params)

    cbh = CallbackHandler(env_params)

    env_params["episode_end_callback"] = cbh.callback_episode_end
    env_params["reward_est_callback"] = cbh.estimate_total_reward
    env_params["reward_est_train_callback"] = cbh.train_total_reward
    env_params["lala"] = cbh.lulu
    create_gym = lambda: aslaug_mod.AslaugEnv(params=env_params)

    env = SubprocVecEnv([create_gym for i in range(n_cpu)])
    g_env = create_gym()
    obs_slicing = g_env.obs_slicing if hasattr(g_env, "obs_slicing") else None

    print("sadfasdfasdf")
    print(dir(env))
    print("IMPOOOOOO")
    cbh.train_total_reward(1,1)

    print(dir(cbh))
    cbh.init_rew_model(obs_slicing)
    print("MAIN", id(cbh))
    cbh.train_total_reward(1,1)
    print(dir(cbh))

    print("SLELELELELE")
    # Prepare curriculum learning
    # if cl is not None:
    #     for cl_entry in cl_list:
    #         env.set_param(cl_entry["param"], cl_entry["start"])

    if pt is None:
        model = PPO2(policy, env, verbose=0,
                     tensorboard_log="data/tb_logs/{}".format(folder_name),
                     policy_kwargs={"obs_slicing": obs_slicing},
                     **learning_params)
    else:
        pfn, pep = pt.split(":")
        model_path = "data/saved_models/{}/aslaug_{}_{}.pkl".format(pfn,
                                                                    version,
                                                                    pep)
        tb_log_path = "data/tb_logs/{}".format(folder_name)
        model = PPO2.load(model_path, env=env, verbose=0,
                          tensorboard_log=tb_log_path,
                          policy_kwargs={"obs_slicing": obs_slicing},
                          **learning_params)
    cbh.set_model(model)
    # Prepare callback
    delta_steps = model.n_batch

    def callback(_locals, _globals):
        global n_steps, model_idx, cl_idx, env, info_idx
        n_steps += delta_steps
        if n_steps / float(n_cp) >= model_idx:
            n_cp_simple = millify(float(model_idx) * float(n_cp), precision=6)
            suffix = "_{}.pkl".format(n_cp_simple)
            cp_name = model_name + suffix
            model.save(dir_path + cp_name)
            model_idx += 1
            data = {"version": version, "model_path": dir_path + cp_name}
            with open('latest.json', 'w') as outfile:
                json.dump(data, outfile)
            print("Stored model at episode {}.".format(n_cp_simple))

        if cl is not None and n_steps / 50000.0 >= cl_idx:
            cl_idx += 1
            for cl_entry in cl_list:
                cl_val = (cl_entry["start"]
                          + (cl_entry["end"]
                             - cl_entry["start"])
                          * n_steps / cl_entry["steps"])
                if cl_val >= min(cl_entry["start"], cl_entry["end"]) \
                        and cl_val <= max(cl_entry["start"], cl_entry["end"]):
                    model.env.env_method("set_param",
                                         cl_entry["param"], cl_val)
                    print("Modifying param {} to {}".format(cl_entry["param"],
                                                            cl_val))

        if n_steps / 5000.0 >= info_idx:
            info_idx += 1
            print("Current frame_rate: {} fps.".format(_locals["fps"]))


    # Start learning
    model.learn(total_timesteps=int(steps), callback=callback)

    # Save model
    model.save(dir_path + model_name + ".pkl")


class EnvScore:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.score_buffer = np.full(self.batch_size, np.nan)

    def add(self, val):
        self.score_buffer = np.roll(self.score_buffer, 1)
        self.score_buffer[0] = val
        return self.is_full()

    def is_full(self):
        return not np.isnan(self.score_buffer).any()

    def get_avg_score(self):
        nansum = np.nansum(self.score_buffer)
        numnonnan = np.count_nonzero(~np.isnan(self.score_buffer))
        return nansum / numnonnan


class CallbackHandler:
    def lulu(self):
        print("Old: ", self.lala)
        self.lala = np.random.randint(100)
        print("New: ", self.lala)
    def __init__(self, learning_params):
        self.lala = 0
        self.lp = learning_params
        self.score = EnvScore(86)
        self.buffer_x = []
        self.buffer_y = []
        self.lock = Lock()
        self.slicing = np.random.randint(100)
        print("REEEEEEEEEEEEEEEEEEEEEEEEEEEEESET")
        print(self.slicing)

        self.tb_cb = keras.callbacks.TensorBoard(log_dir='data/tb_logs/discriminator', histogram_freq=0,
            write_graph=True, write_images=True)

    def init_rew_model(self, obs_slicing):
        with self.lock:
            self.slicing = obs_slicing
            print("SLICING", obs_slicing)
            self.rew_pred_model = self.create_rew_pred_model(obs_slicing)
            self.rew_pred_model.compile(optimizer='adam',
                                        loss='mse',
                                        metrics=['mse'])
    def set_model(self, model):
        with self.lock:
            self.model = model

    def create_rew_pred_model(self, obs_slicing):
        o = obs_slicing
        in_sp = Input(shape=(o[1] - o[0],))
        in_mb = Input(shape=(o[2] - o[1],))
        in_lp = Input(shape=(o[3] - o[2],))
        in_jp = Input(shape=(o[4] - o[3],))
        in_jv = Input(shape=(o[5] - o[4],))
        in_sc1 = Input(shape=(o[6] - o[5],))
        in_sc2 = Input(shape=(o[7] - o[6],))

        s1_0 = Lambda(lambda x: keras.backend.expand_dims(x))(in_sc1)
        s1_1 = Conv1D(4, 11, activation='relu', name="s1_1")(s1_0)
        s1_2 = Conv1D(8, 7, activation='relu', name="s1_2")(s1_1)
        s1_3 = MaxPooling1D(10, 8, name="s1_3")(s1_2)
        s1_4 = Conv1D(8, 7, activation='relu', name="s1_4")(s1_3)
        s1_5 = Conv1D(4, 5, activation='relu', name="s1_5")(s1_4)
        s1_6 = Flatten(name="s1_6")(s1_5)
        s1_7 = Dense(64, activation='relu', name="s1_7")(s1_6)
        s1_out = Dense(32, activation='relu', name="s1_out")(s1_7)

        s2_0 = Lambda(lambda x: keras.backend.expand_dims(x))(in_sc2)
        s2_1 = Conv1D(4, 11, activation='relu', name="s2_1")(s2_0)
        s2_2 = Conv1D(8, 7, activation='relu', name="s2_2")(s2_1)
        s2_3 = MaxPooling1D(10, 8, name="s2_3")(s2_2)
        s2_4 = Conv1D(8, 7, activation='relu', name="s2_4")(s2_3)
        s2_5 = Conv1D(4, 5, activation='relu', name="s2_5")(s2_4)
        s2_6 = Flatten(name="s2_6")(s2_5)
        s2_7 = Dense(64, activation='relu', name="s2_7")(s2_6)
        s2_out = Dense(32, activation='relu', name="s2_out")(s2_7)

        sc_0 = Concatenate()([s1_out, s2_out])
        sc_1 = Dense(32, activation='relu')(sc_0)
        sc_2 = Dense(16, activation='relu')(sc_1)

        t_0 = Concatenate()([sc_2, in_sp, in_mb, in_lp, in_jp, in_jv])
        t_1 = Dense(128, activation='relu')(t_0)
        t_2 = Dense(64, activation='relu')(t_1)
        t_3 = Dense(32, activation='relu')(t_2)
        t_4 = Dense(16, activation='relu')(t_3)
        out = Dense(1, activation='tanh')(t_4)

        return Model(inputs=[in_sp, in_mb, in_lp, in_jp, in_jv, in_sc1, in_sc2], outputs=out)

    def estimate_total_reward(self, init_states):
        if hasattr(self, "rew_pred_model"):
            return self.rew_pred_model.predict(init_states)
        else:
            return [0]

    def train_total_reward(self, init_state, cum_rew):
        with self.lock:
            print("THREEEEED", id(self))
            print("TREN", dir(self))
            print("has slicing", hasattr(self, "slicing"))
            if hasattr(self, "slicing"):
                print(self.slicing)
            return
            self.buffer_x.append(init_state)
            self.buffer_y.append(cum_rew)
            if len(self.buffer_x) >= 4 and hasattr(self, "slicing"):
                print("LALALALA")
                o = self.slicing
                x = np.array(self.buffer_x)
                y = np.array(self.buffer_y)
                in_sp = x[:, o[1] - o[0]]
                in_mb = x[:, o[2] - o[1]]
                in_lp = x[:, o[3] - o[2]]
                in_jp = x[:, o[4] - o[3]]
                in_jv = x[:, o[5] - o[4]]
                in_sc1 = x[:, o[6] - o[5]]
                in_sc2 = x[:, o[7] - o[6]]
                self.rew_pred_model.fit([in_sp, in_mb, in_lp, in_jp, in_jv, in_sc1, in_sc2], y, callbacks=[self.tb_cb])

                self.buffer_x = []
                self.buffer_y = []

    def callback_episode_end(self, env_cb, cum_rew, success):
        if success:
            queue_full = self.score.add(1)
        else:
            queue_full = self.score.add(0)

        if queue_full:
            score = self.score.get_avg_score()

            if score >= self.lp["adr"]["success_threshold"]:
                d_sign = 1
            elif score < self.lp["adr"]["fail_threshold"]:
                d_sign = -1
            else:
                d_sign = 0
            did_modification = False
            if d_sign != 0:
                for el in self.lp["adr"]["adaptions"]:
                    param = el["param"]
                    val_cur = env_cb.get_param(param)
                    delta = (el["end"]-el["start"])/el["steps"]
                    new_val = val_cur + delta * d_sign
                    if el["start"] <= new_val <= el["end"] or \
                            el["end"] <= new_val <= el["start"]:
                        # env_cb.set_param(param, new_val)
                        self.model.env.env_method("set_param",
                                             param, new_val)
                        did_modification = True
                        print("Adapting {} to {}".format(param, new_val))
            if did_modification:
                score.reset()

if __name__ == '__main__':
    main()
