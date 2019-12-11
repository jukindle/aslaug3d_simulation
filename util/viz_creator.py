import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import subprocess
import os


with open('/home/julien/aslaug3d_simulation/data/recordings/aslaug_v2/DD_from_CL/openaigym.video.0.16828.video000002.obs_acts.json', 'r') as f:
    data = json.load(f)


fig, ax = plt.subplots()
xdata, ydata = [], []
ln = plt.bar(["b_x", "b_y", "b_th", "j_1", "j_2"], data['actions'][0])
ax.set_ylim(-1, 1)


files = []

i = 0
last_val = np.array(data['actions'][0])
for x in data['actions']:
    # if i >= 150:
    #     break
    plt.cla()
    val = (np.array(x)-3.0)/3.0
    val = 0.8*last_val + 0.2*val
    plt.bar(["b_x", "b_y", "b_th", "j_1", "j_2"], val)
    last_val = val
    ax.set_ylim(-1, 1)
    fname = '_tmp%03d.png' % i
    i += 1
    print('Saving frame', fname)
    plt.savefig(fname)
    files.append(fname)

print('Making movie animation.mpg - this may take a while')
subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=50 -ovc lavc "
                "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)

# cleanup
for fname in files:
    os.remove(fname)
