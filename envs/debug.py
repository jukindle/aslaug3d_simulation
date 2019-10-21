from aslaug_v1 import AslaugEnv
import numpy as np
import time
import matplotlib.pyplot as plt


# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#
plt.axis([-20,20,-20,20])
plt.ion()
plt.show()
def visualize_obs(obs):
    global env
    in_sp = obs[0:6]
    in_mb = obs[6:9]
    in_lp = obs[9:57]
    in_jp = obs[57:64]
    in_jv = obs[64:71]
    in_sc = obs[71:112]
    n_scans = env.p["sensors"]["lidar"]["n_scans"]
    mag_ang = env.p["sensors"]["lidar"]["ang_mag"]
    angs = ((np.array(range(n_scans))
             - (n_scans-1)/2.0)*2.0/n_scans*mag_ang)
    r_uv = np.vstack((np.cos(angs), np.sin(angs),
                      np.zeros(angs.shape[0])))
    r = r_uv * in_sc
    s = np.ones(r.shape[1])*5
    c = np.ones(r.shape[1])
    plt.clf()
    plt.axis([-20,20,-20,20])
    plt.scatter(r[0, :], r[1, :], s=s, c=c, alpha=0.5)
    plt.scatter([in_sp[2]], [-in_sp[1]], s=[20], c=[1], alpha=0.5)
    plt.pause(0.001)


env = AslaugEnv(gui=False)

env.reset()

nsteps = 100000
ts = time.time()
for i in range(nsteps):
    #print(env.action_space)
    action = np.ones(5, dtype=int)*3
    action[2] = 4
    obs, rew, done, info = env.step(action)
    if i % 10 == 0:
        visualize_obs(obs)
    time.sleep(1/50.0)
te = time.time()
print("RTF: {}".format(nsteps/(te-ts)/50))
