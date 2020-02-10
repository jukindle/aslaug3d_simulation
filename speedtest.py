from stable_baselines import PPO2
import sys
import time
from multiprocessing import Pool
import numpy as np


# Parse inputs
model_path = sys.argv[1]
secs = int(sys.argv[2])
n_models = int(sys.argv[3])


def get_model_fps(model_path, secs):
    model = PPO2.load(model_path)
    obs = model.observation_space.sample()
    model.predict(obs, deterministic=True)

    i = 0
    obs = model.observation_space.sample()
    ts = time.time()
    while time.time() < ts + secs:
        model.predict(obs, deterministic=True)
        i += 1
    return i/secs


# Start tests
if n_models > 1:
    args = [(model_path, secs) for i in range(n_models)]
    pool = Pool(n_models)
    timings = pool.starmap(get_model_fps, args)
    fps = np.average(timings)
else:
    fps = get_model_fps(model_path, secs)

print("Average FPS: {}".format(fps))
