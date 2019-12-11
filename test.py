import time
from envs.aslaug_v5 import AslaugEnv

env = AslaugEnv(gui=True)
env.reset()

for i in range(200):
    env.reset()
    time.sleep(0.5)
# for i in range(1000):
#     env.step(20*[3])
#     time.sleep(0.02)
time.sleep(15)
