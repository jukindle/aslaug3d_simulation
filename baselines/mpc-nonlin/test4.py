import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
import time
import math
import sys
sys.path.append("../..")
from envs.aslaug_v1_cont import AslaugEnv


m = GEKKO(remote=False)

# Prediction params
horizon = 0.125
hz = 15

# Set up approximation poles of arm
l1 = (0.377, 0.074)
l2 = (0.461, -0.104)
l3 = (0.272, 0.0)

# Other params
n_scans = 51

# Calculate sample times
n_steps = int(round(horizon*hz))
m.time = np.linspace(0, horizon, n_steps)


# Define matrix calculation functions
def rotmat(x):
    return (m.cos(x), -m.sin(x), m.sin(x), m.cos(x))


def rotmatT(x):
    return (m.cos(x), m.sin(x), -m.sin(x), m.cos(x))


def cpmat(x):
    return (0, -x, x, 0)


def matvecmul(a, b):
    return (a[0]*b[0]+a[1]*b[1], a[2]*b[0]+a[3]*b[1])


def vecadd(a, b):
    return (a[0]+b[0], a[1]+b[1])


def vecscalarmul(a, b):
    return (a[0]*b, a[1]*b)


# Define function for calculating dynamics of setpoint in EE frame
def v_sp(p_j1, p_j2, v_j1, v_j2, mb_v_x, mb_v_y, mb_v_th):
    s1 = matvecmul(rotmatT(p_j1 + p_j2), (mb_v_x, mb_v_y))
    s2 = matvecmul(cpmat(mb_v_th+v_j1), matvecmul(rotmatT(p_j2), l1))
    s3 = matvecmul(cpmat(mb_v_th+v_j1+v_j2), vecadd(l2, l3))
    v_e = vecadd(s1, vecadd(s2, s3))

    return vecscalarmul(v_e, -1)

def v_lscn(x, y, mb_v_x, mb_v_y, mb_v_th):
    v = vecadd((-mb_v_x, -mb_v_y), matvecmul(cpmat(-mb_v_th), (x, y)))
    return v


# Set up input variables
u_x = m.MV(ub=1.45, lb=-1.45, fixed_initial=False)
u_x.STATUS = 1
u_x.DCOST = 0.0
u_y = m.MV(ub=1.45, lb=-1.45, fixed_initial=False)
u_y.STATUS = 1
u_y.DCOST = 0.0
u_th = m.MV(ub=0.45, lb=-0.45, fixed_initial=False)
u_th.STATUS = 1
u_th.DCOST = 0.0
u_j1 = m.MV(ub=0.7, lb=-0.7, fixed_initial=False)
u_j1.STATUS = 1
u_j1.DCOST = 0.0
u_j2 = m.MV(ub=0.7, lb=-0.7, fixed_initial=False)
u_j2.STATUS = 1
u_j2.DCOST = 0.0

# Setup state variables
sp_x = m.Var()
sp_y = m.Var()
j1_p = m.Var(lb=-2.89, ub=2.89)
j2_p = m.Var(lb=0.05, ub=3.0)
j1_v = m.Var(lb=-0.9, ub=0.9)
j2_v = m.Var(lb=-0.9, ub=0.9)
mb_v_x = m.Var(lb=-0.35, ub=0.35)
mb_v_y = m.Var(lb=-0.35, ub=0.35)
mb_v_th = m.Var(lb=-0.7, ub=0.7)
scn_x = m.Array(m.Var, n_scans)
scn_y = m.Array(m.Var, n_scans)
scn = [(scn_x[i], scn_y[i]) for i in range(len(scn_x))]

# Define functions for state and input processing
def proc_obs(obs):
    pos_sp = np.array([obs[2], -obs[1]])
    mb_vel = np.clip(obs[6:9], [-0.4, -0.4, -0.75], [0.4, 0.4, 0.75])
    linkpos = [0.5, 0, 0, 1.0, 0, 0]
    j_pos = np.clip([obs[45], -obs[46]], [-2.89, 0.05], [2.89, 3.0])
    j_vel = np.clip([obs[47], -obs[48]], [-1.0, -1.0], [1.0, 1.0])
    sc = obs[49:49+n_scans]
    angs = np.linspace(-np.pi, np.pi, n_scans)
    sc_xs, sc_ys = np.cos(angs)*sc, np.sin(angs)*sc
    scf = np.vstack((sc_xs, sc_ys)).T.reshape((-1,))
    op = np.concatenate((pos_sp, mb_vel, j_pos, j_vel, linkpos, scf))

    return op


def proc_input(inp):
    inp[4] = -inp[4]
    return inp


def set_init_states(states):
    st = [sp_x, sp_y, mb_v_x, mb_v_y, mb_v_th, j1_p, j2_p, j1_v, j2_v]
    for i, s in enumerate(st):
        s.value = states[i]
    for i in range(len(scn)):
        scn[i][0].value = states[15+2*i]
        scn[i][1].value = states[16+2*i]


def get_input():
    return [u_x.NEWVAL, u_y.NEWVAL, u_th.NEWVAL, u_j1.NEWVAL, u_j2.NEWVAL]


# Equations: dynamics
m.Equation(mb_v_x.dt() == u_x)
m.Equation(mb_v_y.dt() == u_y)
m.Equation(mb_v_th.dt() == u_th)
m.Equation(j1_v.dt() == u_j1)
m.Equation(j2_v.dt() == u_j2)
m.Equation(j1_p.dt() == j1_v)
m.Equation(j2_p.dt() == j2_v)
m.Equation(sp_x.dt() == v_sp(j1_p, j2_p, j1_v, j2_v,
                             mb_v_x, mb_v_y, mb_v_th)[0])
m.Equation(sp_y.dt() == v_sp(j1_p, j2_p, j1_v, j2_v,
                             mb_v_x, mb_v_y, mb_v_th)[1])
for scn_i in scn:
    m.Equation(scn_i[0].dt() == v_lscn(scn_i[0], scn_i[1], mb_v_x, mb_v_y, mb_v_th)[0])
    m.Equation(scn_i[1].dt() == v_lscn(scn_i[0], scn_i[1], mb_v_x, mb_v_y, mb_v_th)[1])
    m.Equation((scn_i[0]-0.39)**2 + scn_i[1]**2 >= math.sqrt(0.6))
    m.Equation((scn_i[0]-0.54)**2 + scn_i[1]**2 >= math.sqrt(0.6))
    m.Obj(1-abs(m.tanh((scn_i[0]-0.54)**2 + scn_i[1]**2 - math.sqrt(0.6))))
    m.Obj(1-abs(m.tanh((scn_i[0]-0.39)**2 + scn_i[1]**2 - math.sqrt(0.6))))
# Objective Function
p = np.zeros(n_steps)
p[-1] = 1.0
final = m.Param(value=p)

all = m.Param(value=np.ones(n_steps))
print(dir(j2_v))
m.Obj(final*(sp_x**2 + sp_y**2 + 1.0*(j1_v**2 + j2_v**2))+ 0.2/n_steps*((j1_p**2)+(j2_p-1.5)**2))# + 0.1/n_steps*(j1_v**2 + j2_v**2))
# m.Obj(1-abs(m.tanh(10*(j1_p-2.89))))
# m.Obj(1-abs(m.tanh(10*(j1_p+2.89))))
# m.Obj(1-abs(m.tanh(10*(j2_p-0.05))))
# m.Obj(1-abs(m.tanh(10*(j2_p+3.0))))
# m.Obj(1-abs(m.tanh(10*(j1_v+0.9))))
# m.Obj(1-abs(m.tanh(10*(j2_v+0.9))))
# m.Obj(1-abs(m.tanh(10*(j2_v-0.9))))
# m.Obj(1-abs(m.tanh(10*(j1_v-0.9))))
# m.Obj(1-abs(m.tanh(10*(mb_v_x-0.35))))
# m.Obj(1-abs(m.tanh(10*(mb_v_x+0.35))))
# m.Obj(1-abs(m.tanh(10*(mb_v_y-0.35))))
# m.Obj(1-abs(m.tanh(10*(mb_v_y+0.35))))
# m.Obj(1-abs(m.tanh(10*(mb_v_th-0.7))))
# m.Obj(1-abs(m.tanh(10*(mb_v_th+0.7))))
# Configure optimizer
m.options.IMODE = 6
m.options.MAX_ITER = 1000
m.options.WEB = 0
m.options.NODES = 2
m.options.CV_TYPE = 2
m.options.MAX_TIME = 1.0
m.options.SOLVER = 3

# Set up env
env = AslaugEnv(gui=True)
obs = proc_obs(env.reset())

for i in range(1000):
    ts = time.time()
    set_init_states(obs)
    try:
        m.solve(disp=False, debug=True, GUI=False)
    except Exception:
        print("INFEASIBLE, IGNORING")
    inp = get_input()
    inp = proc_input(inp)
    #print(inp)
    for i in range(int((50/hz))):
        obs, _, _, _ = env.step(np.array(inp))
    obs = proc_obs(obs)
    te = time.time()
    if te-ts < 50/hz*0.02:
        time.sleep(50/hz*0.02 - (te-ts))






#
#
# # print('Objective: ' + str(1-x1[-1]-x2[-1]))
#
# plt.figure(1)
#
# plt.subplot(2,1,1)
# plt.plot(m.time, sp_x.value,'k:',LineWidth=2,label=r'$sp_x$')
# plt.plot(m.time, sp_y.value,'b-',LineWidth=2,label=r'$sp_y$')
# plt.ylabel('Value')
# plt.legend(loc='best')
#
# plt.subplot(2,1,2)
# plt.plot(m.time, u_j1.value,'r-',LineWidth=2,label=r'$u_j1$')
# plt.legend(loc='best')
# plt.xlabel('Time')
# plt.ylabel('Value')
#
# plt.show()
