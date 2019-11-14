import keyboard #Using module keyboard
from envs.aslaug_v2 import AslaugEnv
import numpy as np
import time

env = AslaugEnv(gui=True)

env.reset()

nsteps = 100000
ts = time.time()
for i in range(nsteps):
    print(env.obs_slicing)
    action = np.ones(5, dtype=int)*3
    if keyboard.is_pressed('up'):
        action[0] = 6
    if keyboard.is_pressed('down'):
        action[0] = 0
    if keyboard.is_pressed('left'):
        action[1] = 6
    if keyboard.is_pressed('right'):
        action[1] = 0
    if keyboard.is_pressed('page up'):
        action[2] = 6
    if keyboard.is_pressed('page down'):
        action[2] = 0
    obs, rew, done, info = env.step(action)
    # print(obs.shape)
    #print(rew)
    print(done)
    print(info)
    time.sleep(1/50.0)
te = time.time()
print("RTF: {}".format(nsteps/(te-ts)/50))
