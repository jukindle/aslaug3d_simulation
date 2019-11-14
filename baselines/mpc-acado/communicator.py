import os
os.chdir("../..")
print(os.getcwd())
import sys
sys.path.append('')
from envs.aslaug_v1_cont import AslaugEnv
env = None
def setup():
    global env, obs
    env = AslaugEnv(gui=True)
    os.chdir("baselines/mpc-acado")
    obs = env.reset()
def get_obs():
    global obs
    return obs.tolist()

def step(inp):
    global env

    obs, r, d, _ = env.step(inp)
    return obs.tolist()

def close():
    env.close()
