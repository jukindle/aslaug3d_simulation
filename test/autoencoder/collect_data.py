import numpy as np
import sys
sys.path.append("../..")
from envs.aslaug_v1 import AslaugEnv
from stable_baselines import PPO2

n_steps = 100000


model = PPO2.load("../../data/saved_models/v1_hold_largtol_cont/aslaug_v1.pkl")

env = AslaugEnv(gui=False)



data = np.zeros((n_steps, env.observation_space.shape[0]))
episode_done = True
for n in range(n_steps):
    if episode_done:
        obs = env.reset()
        episode_done = False
    else:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, episode_done, _ = env.step(action)
    data[n, :] = obs

np.save("data.npy", data)
print("Collected {} observations!".format(n_steps))
print("Shape of data: {}".format(data.shape))
