import time
import sys
sys.path.append("../..")
from envs.aslaug_v1_cont import AslaugEnv
import numpy as np


env = AslaugEnv(gui=True)

obs = env.reset()


while True:
    ac = [0.0, 0.0, 1.0, 0.0, 0.0]
    obs, _, _, _ = env.step(ac)
    print(np.round(obs[0:3], 3))
    time.sleep(100.1)
