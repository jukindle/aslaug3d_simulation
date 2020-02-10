from stable_baselines import PPO2
import sys
import time


model = PPO2.load(sys.argv[1])
exec_time = int(sys.argv[2])

i = 0
ts = time.time()
while time.time() < ts + exec_time:
    obs = model.observation_space.sample()
    act = model.predict(obs, deterministic=True)
    i = i + 1

print("Average FPS: {}".format(i / exec_time))
