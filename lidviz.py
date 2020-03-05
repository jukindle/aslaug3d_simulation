import numpy as np
import matplotlib.pyplot as plt
N = 201
max_ang = 2.09
scan_range = 5.0


def magic(N, max_ang, scan_range, r):
    angs = ((np.array(range(N))
             - (N-1)/2.0)*2.0/N*max_ang)

    r_uv = np.vstack((np.cos(angs), np.sin(angs),
                      np.zeros(angs.shape[0])))
    r_from = r_uv * 0.1
    r_to = r_uv * scan_range



    return r*r_uv

p1 = magic(201, max_ang, scan_range, np.linspace(0.1, 5.0, 201))
p2 = magic(101, max_ang, scan_range, np.linspace(5.0, 0.1, 101))
plt.plot(p1[0], p1[1], 'ro', markersize=1)
plt.plot(p2[0], p2[1], 'bo', markersize=1)
plt.axis([-5, 5, -5, 5])
plt.show()
