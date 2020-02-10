from stable_baselines import PPO2
import sys
import time


exec_time = int(sys.argv[2])

n_models = 1
if len(sys.argv) > 3:
    n_models = int(sys.argv[3])

models = [PPO2.load(sys.argv[1]) for i in range(n_models)]

def step_all(models):
    obs = models[0].observation_space.sample()
    for model in models:
        model.predict(obs, deterministic=True)


i = 0
ts = time.time()
while time.time() < ts + exec_time:
    step_all(models)
    i = i + 1

print("Average FPS: {}".format(i / exec_time / n_models))
