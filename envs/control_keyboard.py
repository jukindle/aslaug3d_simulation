import keyboard #Using module keyboard
from aslaug_v1lw import AslaugEnv
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
        action[0] = 4
    if keyboard.is_pressed('down'):
        action[0] = 2
    if keyboard.is_pressed('left'):
        action[1] = 4
    if keyboard.is_pressed('right'):
        action[1] = 2
    if keyboard.is_pressed('page up'):
        action[2] = 4
    if keyboard.is_pressed('page down'):
        action[2] = 2
    obs, rew, done, info = env.step(action)
    print(obs.shape)
    #print(rew)
    print(done)
    print(info)
    time.sleep(1/50.0)
te = time.time()
print("RTF: {}".format(nsteps/(te-ts)/50))



while True:  #making a loop
    try:  #used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('up'): #if key 'up' is pressed.You can use right,left,up,down and others
            print('You Pressed A Key!')
            break #finishing the loop
        else:
            pass
    except:
        break  #if user pressed other than the given key the loop will break
