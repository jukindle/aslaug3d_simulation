import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import traceback

rootdir = '/home/julien/tb_logs'


fig = plt.figure()
plot = fig.add_subplot(111)

annot = plot.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))

def on_plot_hover(event):
    # Iterating over each data member plotted
    for curve in plot.get_lines():
        # Searching which data member corresponds to current mouse position
        if curve.contains(event)[0]:
            annot.xy = [event.xdata, event.ydata]
            annot.set_text(curve.get_gid())
            fig.canvas.draw_idle()

def log_to_numpy(file_path):
    data = []
    for e in tf.train.summary_iterator(file_path):
        try:
            for v in e.summary.value:
                if v.tag == "episode_reward":
                    data.append([e.step, v.simple_value])
        except Exception:
            pass
    return np.array(data)

def smooth_data(data_in):
    data = data_in.copy()
    filter = np.ones(200)/200.0
    data[:, 1] = np.convolve(data[:, 1], filter, 'same')
    return data

print("Scanning through files...")
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(file)
        if not file.startswith("events.out.tfevents"):
            continue
        dirname = subdir[len(rootdir):]
        file_path = os.path.join(subdir, file)
        try:
            data = log_to_numpy(file_path)
            data = smooth_data(data)
            plot.plot(data[:, 0], data[:, 1], gid=dirname)
        except Exception as e:
            print("Error on {}".format(dirname))
            print(e)
            traceback.print_exc()
print("Done scanning!")



fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

plt.show()
