from envs.aslaug_v2 import AslaugEnv
import time

env = AslaugEnv(gui=True)


while True:
    env.reset()
    time.sleep(2.0)

time.sleep(100)
