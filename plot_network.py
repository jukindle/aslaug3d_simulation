from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt


def cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(positions, sizes=None, colors=None, alpha=1.0, **kwargs):
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"]*len(positions)
    if not isinstance(sizes, (list, np.ndarray)):
        sizes = [(1, 1, 1)]*len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))

    x = Poly3DCollection(np.concatenate(g), **kwargs)
    x.set_alpha(alpha)
    x.set_facecolor(np.repeat(colors, 6))
    return x


class Drawer:
    def __init__(self, debug=True, text_dis=20, font_size=20):
        self.fontsize = font_size
        self.text_dis = text_dis
        self.mins = np.zeros(3)
        self.maxs = np.zeros(3)
        self.debug = debug
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_aspect('equal')

    def add_layer(self, pos, sy, sz, col='blue', text="", text_pos='bottom', textdis=20):

        pos_a = np.array([pos[0]-1, pos[1]-sy/2.0, pos[2]-sz/2.0])
        size = np.array([1, sy, sz])
        pc = plotCubeAt2([pos_a], [size], colors=[col], alpha=0.2, edgecolor='k')
        self.ax.add_collection3d(pc)

        pos_mm = np.array(pos) - np.array((0.5, 0, 0))
        self.mins = np.min((self.mins, pos_mm-size/2.0), axis=0)
        self.maxs = np.max((self.maxs, pos_mm+size/2.0), axis=0)

        if text_pos == 'bottom':
            self.ax.text(pos_a[0]+0.5, pos_a[1], pos_a[2]-textdis, text, color='black',fontsize=self.fontsize, horizontalalignment='center', verticalalignment='top')
        if text_pos == 'left':
            self.ax.text(pos_a[0]-1-textdis, 0, pos_a[2], text, color='black',fontsize=self.fontsize, horizontalalignment='right', verticalalignment='center')

    def plot(self):
        self.ax.set_aspect('auto')
        plt.axis('off')
        self.ax.set_xlim([self.mins[0], self.maxs[0]])
        self.ax.set_ylim([self.mins[1], self.maxs[1]])
        self.ax.set_zlim([self.mins[2], self.maxs[2]])


        if self.debug:
            self.ax.quiver(0,0,0,0,1,0,color='red')
            self.ax.quiver(0,0,0,1,0,0,color='green')
            self.ax.quiver(0,0,0,0,0,1,color='blue')

        plt.show()


class Layer:
    def __init__(self, sy, sz, col='blue', textdis=20):
        self.text_dis = textdis
        self.sy = sy
        self.sz = sz
        self.col = col


class DoubleLayer:
    def __init__(self, sy, sz, col='blue', textdis=20):
        self.text_dis = textdis
        self.sy = sy
        self.sz = sz
        self.col = col


class InputLayer:
    def __init__(self, input_names, input_sizes, col='blue', text_dis=0):
        self.text_dis = text_dis
        self.input_names = input_names
        self.input_sizes = input_sizes
        self.col = col


class Path:
    def __init__(self):
        self.layers = []
        self.paths_before = []
        self.path_next = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def connect_to(self, path_next):
        self.path_next = path_next
        path_next.paths_before.append(self)


class Arranger:
    def __init__(self, layer_dis, path_dis=100):
        self.layer_dis = layer_dis
        self.path_dis = path_dis
        self.dw = Drawer()

    def draw(self, last_path, pos=[0, 0, 0], fac=1.0):
        layers = list(last_path.layers)
        layers.reverse()

        pos_c = np.array(pos)
        for layer in layers:
            if isinstance(layer, Layer):
                if layer.sz == 1:
                    txt = layer.sy
                elif layer.sy == 1:
                    txt = layer.sz
                else:
                    txt = "({},{})".format(layer.sz, layer.sy)
                self.dw.add_layer(pos_c, layer.sy, layer.sz, col=layer.col, text=txt, text_pos='bottom', textdis=layer.text_dis)


            if isinstance(layer, DoubleLayer):
                sel = [40, -80]
                if layer.sz > layer.sy:
                    sel = [80, -160]
                if layer.sz == 1:
                    txt = layer.sy
                elif layer.sy == 1:
                    txt = layer.sz
                else:
                    txt = "({},{})".format(layer.sz, layer.sy)
                pos_cc = list(pos_c)
                pos_cc[2] += sel[0]
                self.dw.add_layer(pos_cc, layer.sy, layer.sz, col=layer.col, text='', text_pos='bottom')
                pos_cc[2] += sel[1]
                self.dw.add_layer(pos_cc, layer.sy, layer.sz, col=layer.col, text=txt, text_pos='bottom', textdis=layer.text_dis)

            if isinstance(layer, InputLayer):
                n_l = len(layer.input_names)
                sel = []
                if n_l == 2:
                    sel = [40, -80]

                if n_l == 5:
                    sel = [40, -20, -20, -20, -20]
                for i in range(len(sel)):
                    pos_c[2] += sel[i]*fac
                    txt = layer.input_names[i]
                    sy, sz = layer.input_sizes[i]
                    self.dw.add_layer(pos_c, sy, sz, col=layer.col, text=txt, text_pos='left', textdis=layer.text_dis)
            pos_c[0] -= 1 + self.layer_dis

        n_layers = len(last_path.paths_before)
        if n_layers == 0:
            sel = []
        elif n_layers == 1:
            sel = [0]
        elif n_layers == 2:
            sel = [self.path_dis/2.0, -self.path_dis]
        elif n_layers == 3:
            sel = [self.path_dis, -self.path_dis, -self.path_dis]
        else:
            print("NOT IMPLEMENTED SO MANY PATHS YET")
            return
        for idx, layer in enumerate(last_path.paths_before):
            pos_c[2] += sel[idx]
            self.draw(layer, pos_c)

    def plot(self):
        self.dw.plot()


col_dense = 'blue'
col_conv = 'red'
col_pool = 'black'
col_concat = 'green'
col_input = 'white'
col_flatten = 'white'

scan_input = Path()
scan_input.add_layer(InputLayer(['scan_front', 'scan_rear'], [(201, 1), (201, 1)], col_input, text_dis=2))

scan_block = Path()
scan_input.connect_to(scan_block)
scan_block.add_layer(DoubleLayer(191, 2, col_conv))  # Conv1D(2, 11)
scan_block.add_layer(DoubleLayer(185, 4, col_conv, 45))  # Conv1D(4, 7)
scan_block.add_layer(DoubleLayer(183, 8, col_conv))  # Conv1D(8, 3)
scan_block.add_layer(DoubleLayer(61, 8, col_pool))  # MaxPooling1D(3, 3)
scan_block.add_layer(DoubleLayer(55, 8, col_conv, 45))  # Conv1D(8, 7)
scan_block.add_layer(DoubleLayer(51, 8, col_conv))  # Conv1D(8, 5)
scan_block.add_layer(DoubleLayer(17, 8, col_pool, 10))  # MaxPooling1D(3, 3)
scan_block.add_layer(DoubleLayer(15, 8, col_conv, 30))  # Conv1D(8, 3)
scan_block.add_layer(DoubleLayer(1, 120, col_flatten))  # Flatten()
scan_block.add_layer(DoubleLayer(1, 128, col_dense))  # Dense(128)
scan_block.add_layer(DoubleLayer(1, 64, col_dense))  # Dense(64)
scan_block.add_layer(DoubleLayer(1, 64, col_dense))  # Dense(64)

# Combination of both scans
scan_block.add_layer(Layer(1, 128, col_dense))  # Dense
scan_block.add_layer(Layer(1, 128, col_dense))  # Dense
scan_block.add_layer(Layer(1, 64, col_dense))  # Dense
scan_block.add_layer(Layer(1, 64, col_dense))  # Dense

comb_input = Path()
comb_input.add_layer(InputLayer(['setpoint', 'base vels', 'ee pos', 'joint pos', 'joint vels'], [(1, 3), (1, 3), (1, 3), (1, 2), (1, 2)], col_input))

combination_block = Path()
comb_input.connect_to(combination_block)
# Combination block
combination_block.add_layer(Layer(1, 77, col_concat))  # Concat
combination_block.add_layer(Layer(1, 512, col_dense))  # Dense(512)
combination_block.add_layer(Layer(1, 256, col_dense))  # Dense(256)
combination_block.add_layer(Layer(1, 256, col_dense))  # Dense(256)
combination_block.add_layer(Layer(1, 128, col_dense))  # Dense(128)
combination_block.add_layer(Layer(1, 128, col_dense))  # Dense(128)
combination_block.add_layer(Layer(1, 64, col_dense))  # Dense(64)
combination_block.add_layer(Layer(1, 64, col_dense))  # Dense(64)
combination_block.add_layer(Layer(1, 32, col_dense))  # Dense(32)


ac_block_vf = Path()
ac_block_vf.add_layer(Layer(1, 64))
ac_block_vf.add_layer(Layer(1, 16))
ac_block_vf.add_layer(Layer(1, 1))
ac_block_p = Path()
ac_block_p.add_layer(Layer(1, 64))
ac_block_p.add_layer(Layer(1, 32))

scan_block.connect_to(combination_block)
combination_block.connect_to(ac_block_vf)

agr = Arranger(3, 200)
agr.draw(ac_block_vf)
agr.plot()
