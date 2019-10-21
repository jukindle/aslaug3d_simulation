from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from millify import millify
import numpy as np

from importlib import import_module
import argparse
import os
import shutil
import json
import random

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", help="Define # steps for learning.",
                        default=10e6)
    parser.add_argument("-n", "--n_cpu", help="Define # processes to use.",
                        default=32)
    parser.add_argument("-v", "--version", help="Define version of env to use.")
    parser.add_argument("-p", "--policy", help="Define policy to use (path).",
                        default="stable_baselines.common.policies.MlpPolicy")
    parser.add_argument("-cp", "--check_point", help="# steps in between checkp..",
                        default=500e3)
    parser.add_argument("-f", "--folder", help="Name the folder",
                        default="None")
    parser.add_argument("-de", "--default_env", help="Use default gym env.",
                        default="None")
    parser.add_argument("-cl", "--curriculum_learning", action='append',
                        help="Enable curriculum learning. Example to adjust \
                        parameter p1 from 1 to 5 in 3M steps: -cl p1:1:5:3e6.")
    args = parser.parse_args()

    n_cpu, version = int(float(args.n_cpu)), args.version
    steps = int(float(args.steps))
    policy_arg = args.policy
    n_cp = args.check_point
    folder_name = args.folder
    cl = args.curriculum_learning

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
    #os.system("printf '\e]0;{}\7\n'".format(folder_name))
    #os.system("tmux rename-session {}-{}".format(version, folder_name))

    dir_path = "/cluster/scratch/jkindle/data/saved_models/{}/".format(folder_name)
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
        m = (lr_max-lr_min)/(a-b)
        c = lr_max-m*a

        return lambda x: np.min([lr_max, np.max([lr_min, m*x + c])])


    # Prepare model and env
    aslaug_mod = import_module("envs." + model_name)
    shutil.copy("envs/{}.py".format(model_name), dir_path + model_name + ".py")
    create_gym = lambda: aslaug_mod.AslaugEnv()

    env = SubprocVecEnv([create_gym for i in range(n_cpu)])
    env_ghost = create_gym()
    if cl is not None:
        for cl_entry in cl_list:
            env.set_attr(cl_entry["param"], cl_entry["start"])








    hyperparams_ranges = {
        "nminibatches": [4, 8, 16, 32],
        "n_steps": [16, 32, 64, 128, 256, 512, 1024, 2048],
        "gamma": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
        "learning_rate": (1e-5, 1e-3),
        "ent_coef": (0.00000001, 0.01),
        "cliprange": [0.1, 0.2, 0.3, 0.4],
        "noptepochs": [1, 5, 10, 20, 30, 50],
        "lam": [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    }


    def select_key_randomly(key, hyperparams_ranges):
        ranges = hyperparams_ranges[key]
        if type(ranges) is tuple:
            return np.random.uniform(ranges[0], ranges[1])
        else:
            return random.sample(ranges, 1)[0]

    params = {}
    for key in hyperparams_ranges:
        params[key] = select_key_randomly(key, hyperparams_ranges)

    params["cliprange_vf"] = -1
    learning_params = params
    # learning_params = {"n_steps": 128, "nminibatches": 8,  "lam": 0.95,
    #                    "gamma": 0.9, "noptepochs": 4, "ent_coef": .001,
    #                    "learning_rate": [1e-3, 0.01e-4, 0.9, 0.05],
    #                    "cliprange": [0.3, 0.1, 0.9, 0.35],
    #                    "cliprange_vf": -1}

    # Save learning params to file
    params_file = "/cluster/scratch/jkindle/data/saved_models/{}/learning_params.json".format(folder_name)
    with open(params_file, 'w') as outfile:
        json.dump(learning_params, outfile)
    # Copy policy to models folder
    shutil.copy(policy_mod.__file__,
                "/cluster/scratch/jkindle/data/saved_models/{}/{}.py".format(folder_name, policy_name))

    if type(learning_params["learning_rate"]) in [list, tuple]:
        lr_params = learning_params["learning_rate"]
        learning_params["learning_rate"] = create_custom_lr(*lr_params)
    if type(learning_params["cliprange"]) in [list, tuple]:
        lr_params = learning_params["cliprange"]
        learning_params["cliprange"] = create_custom_lr(*lr_params)


    g_env = create_gym()
    obs_slicing = g_env.obs_slicing if hasattr(g_env, "obs_slicing") else None
    model = PPO2(policy, env, verbose=0,
                 tensorboard_log="/cluster/scratch/jkindle/data/tb_logs/{}".format(folder_name),
                 policy_kwargs={"obs_slicing": obs_slicing},
                 **learning_params)


    # Prepare callback
    n_steps = 0
    delta_steps = model.n_batch
    model_idx = 0
    cl_idx = 0
    info_idx = 0


    def callback(_locals, _globals):
        global n_steps, model_idx, cl_idx, env, info_idx
        n_steps += delta_steps
        if n_steps / float(n_cp) >= model_idx:
            n_cp_simple = millify(float(model_idx)*float(n_cp), precision=6)
            suffix = "_{}.pkl".format(n_cp_simple)
            cp_name = model_name + suffix
            model.save(dir_path + cp_name)
            model_idx += 1
            data = {"version": version, "model_path": dir_path+cp_name}
            with open('latest.json', 'w') as outfile:
                json.dump(data, outfile)
            print("Stored model at episode {}.".format(n_cp_simple))

        if cl is not None and n_steps / 50000.0 >= cl_idx:
            cl_idx += 1
            for cl_entry in cl_list:
                cl_val = (cl_entry["start"]
                          + (cl_entry["end"]
                             - cl_entry["start"])*n_steps/cl_entry["steps"])
                if cl_val >= min(cl_entry["start"], cl_entry["end"]) \
                        and cl_val <= max(cl_entry["start"], cl_entry["end"]):
                    env.set_attr(cl_entry["param"], cl_val)
                    print("Modifying param {} to {}".format(cl_entry["param"],
                                                            cl_val))

        if n_steps / 5000.0 >= info_idx:
            info_idx += 1
            print("Current frame_rate: {} fps.".format(_locals["fps"]))


    # Start learning
    print("=============={}==============".format(folder_name))
    model.learn(total_timesteps=int(steps), callback=callback)

    # Save model
    model.save(dir_path + model_name + ".pkl")
