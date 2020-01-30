import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


filename, smoothing = sys.argv[1], float(sys.argv[2])
name = filename.split('/')[-1][:-4]
path = '/'.join(filename.split('/')[:-1])
df = pd.read_csv(filename, delimiter=',', header=1)

data = np.array(df)
vals = data[:, 2]
vals[vals <= -100] = -100
k = np.ones(int(vals.shape[0]*smoothing))/int(vals.shape[0]*smoothing)


def smooth(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array([smoothed])


filt = smooth(vals, smoothing).T
print(filt.shape)
print(data.shape)
data = np.append(data, filt, axis=1)


np.savetxt("{}/{}_out.csv".format(path, name), data, delimiter=",", header="Wall time,Step,Value,Smoothed", comments='', fmt='%1.5f')

plt.plot(data[:, 1], vals,alpha=0.1 )
plt.plot(data[:, 1], filt)
axes = plt.gca()
# axes.set_ylim([-100, 100])
plt.show()
