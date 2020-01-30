from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from millify import millify
import numpy as np

from importlib import import_module
import argparse
import os
import shutil
import json
from util.tb_logging import Logger

# Prepare callback global parameters
n_steps = 0
model_idx = 0
cl_idx = 0
info_idx = 0
ADR_idx = 1
spwnrng = 1.5
last_adr_idx = 0
logger = None


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
    parser.add_argument("-use_dis", "--use_discriminator", default=0,
                        help="Specify if discriminator shall be used.")
    args = parser.parse_args()

    n_cpu, version = int(float(args.n_cpu)), args.version
    steps = int(float(args.steps))
    policy_arg = args.policy
    n_cp = args.check_point
    folder_name = args.folder
    cl = args.curriculum_learning
    pt = args.proceed_training
    use_dis = int(args.use_discriminator)

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
    os.system("tmux set -g status-left-length 50")
    os.system("tmux set -g status-left \"{}: {}    \"".format(version, folder_name))
    os.system("tmux set -g status-right \"\"")



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


    create_gym = lambda: aslaug_mod.AslaugEnv(params=env_params)
    env = SubprocVecEnv([create_gym for i in range(n_cpu)])
    g_env = create_gym()
    obs_slicing = g_env.obs_slicing if hasattr(g_env, "obs_slicing") else None
    lidar_calib = np.array(g_env.get_lidar_calibration())
    np.save("data/saved_models/{}/lidar_calib.npy".format(folder_name), lidar_calib)

    if use_dis:
        import threading
        from discriminator import Discriminator
        discr = Discriminator(env_params["discriminator"], obs_slicing,
                              g_env.observation_space)
        thread = threading.Thread(target=discr.main)
        thread.start()
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
    # Prepare callback
    delta_steps = model.n_batch

    def callback(_locals, _globals):
        global n_steps, model_idx, cl_idx, env, info_idx, ADR_idx, spwnrng, last_adr_idx, logger
        n_cp_simple = 0
        n_steps += delta_steps
        if logger is None:
            ppo_id = 1
            ppo_path = 'data/tb_logs/{}/PPO2_{}'.format(folder_name, ppo_id+1)
            while os.path.exists(ppo_path):
                ppo_id += 1
                ppo_path = 'data/tb_logs/{}/PPO2_{}'.format(folder_name, ppo_id+1)
            ppo_path = 'data/tb_logs/{}/PPO2_{}/addons'.format(folder_name, ppo_id)
            logger = Logger(ppo_path)
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
            logger.log_scalar('metrics/success_rate', np.average(model.env.env_method("get_success_rate")), n_steps)

            # os.system("tmux set -g status-right \"Steps {} / {} | ADR {} | FPS {}\"".format(n_cp_simple, millify(float(steps), precision=6), spwnrng, _locals["fps"]))

        if n_steps / 25000.0 >= ADR_idx and len(env_params['adr']['adaptions']) > 0:
            ADR_idx += 1
            avg = np.average(model.env.env_method("get_success_rate"))
            print("Average success rate: {}".format(avg))
            for adaption in env_params['adr']['adaptions']:
                val = np.average(model.env.env_method("get_param", adaption['param']))
                logger.log_scalar('ADR/{}'.format(adaption['param']), val, n_steps)
            if avg >= env_params['adr']['success_threshold']:
                last_adr_idx = np.random.randint(len(env_params['adr']['adaptions']))
                lp = env_params['adr']['adaptions'][last_adr_idx]
                val = np.average(model.env.env_method("get_param", lp['param']))
                dval = +(lp['end']-lp['start'])/lp['steps']
                val = max(min(lp['end'], lp['start']), min(max(lp['end'], lp['start']), val + dval))
                print("Setting {} to {}(+)".format(lp['param'], val))
                model.env.env_method("set_param", lp['param']
                                     , val)
            if avg <= env_params['adr']['fail_threshold']:
                lp = env_params['adr']['adaptions'][last_adr_idx]
                val = np.average(model.env.env_method("get_param", lp['param']))
                dval = -(lp['end']-lp['start'])/lp['steps']
                val = max(min(lp['end'], lp['start']), min(max(lp['end'], lp['start']), val + dval))
                print("Setting {} to {}(-)".format(lp['param'], val))
                model.env.env_method("set_param", lp['param']
                                     , val)

    # Print number of trainable weights
    n_els = np.sum([x.shape.num_elements()*x.trainable
                    for x in model.get_parameter_list()])
    print("Number of trainable weights: {}".format(n_els))
    # Start learning
    model.learn(total_timesteps=int(steps), callback=callback)

    # Save model
    model.save(dir_path + model_name + ".pkl")



if __name__ == '__main__':
    main()
