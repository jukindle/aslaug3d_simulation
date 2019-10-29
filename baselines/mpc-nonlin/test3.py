import osqp
import numpy as np
import sympy as sp
from scipy import sparse
import time
import sys
sys.path.append("../..")
from envs.aslaug_v1_cont import AslaugEnv

# Setup system dynamics
l1 = sp.Matrix([[0.2], [0.0]])
l2 = sp.Matrix([[0.2], [0.0]])
l3 = sp.Matrix([[0.1], [0.0]])


def rotmat(a):
    return sp.Matrix([[sp.cos(a), -sp.sin(a)], [sp.sin(a), sp.cos(a)]])


vars = sp.symbols('x_0:97')
inps = sp.symbols('u_0:5')
r_t_ee = rotmat(vars[5]) * (l1 + rotmat(vars[6]) * (l2 + l3))
J = r_t_ee.jacobian(vars[5:7])
r_v_ee = J*sp.Matrix(vars[7:9])
r_v_wee = (sp.Matrix(vars[2:4])
           + sp.Matrix([[-vars[4], 0.0], [0.0, vars[4]]])*r_v_ee)
ee_v_w = -rotmat(vars[5] + vars[6])*r_v_wee

sp_n = sp.Matrix(vars[0:2]) + ee_v_w*0.02  # NOTE: or plus?
vmb_n = sp.Matrix(vars[2:5]) + sp.Matrix(inps[0:3])
j_n = sp.Matrix(vars[5:7]) + sp.Matrix(vars[7:9])*0.02
jv_n = sp.Matrix(vars[7:9]) + sp.Matrix(inps[3:5])

r_l1_p = rotmat(vars[5])*l1/2.0
r_l2_p = rotmat(vars[5])*(l1 + rotmat(vars[6])*l2/2.0)

ls_n = sp.Matrix((r_l1_p, vars[5:6], r_l2_p, [vars[5] + vars[6]]))

rotmat_huge_diag = [rotmat(0.02*vars[4]).T for x in range(41)]
shifts_huge = sp.Matrix([0.02*vars[2+(x % 2)] for x in range(2*41)])
sc_n = sp.diag(*rotmat_huge_diag)*(sp.Matrix(vars[15:97]) - shifts_huge)

state_new = sp.Matrix([sp_n, vmb_n, j_n, jv_n, ls_n, sc_n])
A_f = sp.lambdify((vars, inps), state_new.jacobian(vars), 'scipy')
B_f = sp.lambdify((vars, inps), state_new.jacobian(inps), 'scipy')


# Constraints
umin = np.array([-0.03, -0.03, -0.01, -0.015, -0.015])
umax = np.array([+0.03, +0.03, +0.01, +0.015, +0.015])

xmin = np.array([-np.inf, -np.inf,
                 -0.4, -0.4, -0.75,
                 -2.89, 0.05,
                 -1.0, -1.0,
                 -np.inf, -np.inf, -np.inf,
                 -np.inf, -np.inf, -np.inf]
                + 82*[-np.inf])
xmax = np.array([+np.inf, +np.inf,
                 +0.4, +0.4, +0.75,
                 +2.89, 3.0,
                 +1.0, +1.0,
                 +np.inf, +np.inf, +np.inf,
                 +np.inf, +np.inf, +np.inf]
                + 82*[+np.inf])
# Objective function
Q = sparse.diags([1, 1] + 95*[0.0])
QN = Q
R = 0.0*sparse.eye(5)

# Initial and reference states
x0 = np.array([6, 0, 0.1, 0.1, 0.01, 0.5, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]
              + [3*np.sin(x) if x % 2 else 3*np.cos(x) for x in np.linspace(-np.pi/2, np.pi/2, 82)])

xr = x0

# Obtain taylor approximation
Ad = A_f(x0, [0, 0, 0, 0, 0])
Bd = B_f(x0, [0, 0, 0, 0, 0])
[nx, nu] = Bd.shape

# Prediction horizon
N = 25




def calc_matrices(Ad, Bd, x0, xr):
    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                           sparse.kron(sparse.eye(N), R)], format='csc')
    # - linear objective
    q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                   np.zeros(N*nu)])

    # - input and state constraints
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    leq = np.hstack([-x0, np.zeros(N*nx)])
    ueq = leq
    lb = np.hstack([leq, lineq])
    ub = np.hstack([ueq, uineq])
    # - linear dynamics
    Ax = sparse.kron(sparse.eye(N+1),
                     -sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]),
                     Bd)
    Aeq = sparse.hstack([Ax, Bu])

    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format='csc')

    return A, lb, ub, q, P


A, lb, ub, q, P = calc_matrices(Ad, Bd, x0, xr)

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, lb, ub, warm_start=True)


# Process observation to state
def proc_obs(obs):
    pos_sp = np.array([obs[2], -obs[1]])
    mb_vel = obs[6:9]
    linkpos = [0.5, 0, 0, 1.0, 0, 0]
    j_pos = [obs[45], -obs[46]]
    j_vel = obs[47:49]
    sc = obs[49:90]
    angs = np.linspace(-np.pi/2, np.pi/2, 41)
    sc_xs, sc_ys = np.cos(angs)*sc, np.sin(angs)*sc
    scf = np.vstack((sc_xs, sc_ys)).T.reshape((-1,))
    op = np.concatenate((pos_sp, mb_vel, j_pos, j_vel, linkpos, scf))
    op = np.clip(op, xmin, xmax)
    return op
np.set_printoptions(threshold=np.inf, precision=3)

# Setup simulation
env = AslaugEnv(gui=True)
obs = proc_obs(env.reset())
x0 = np.array(obs)
# print(x0)
# print("=================")
#
# obs = [-2.554e-01,  1.065e+00,  3.858e-02,  2.995e-01, -2.570e-10, -1.437e+00,
#  -1.368e+00,1.407e-03,1.085e-03,5.000e-01,0.000e+00,0.000e+00,
# 1.000e+00,0.000e+00,0.000e+00,2.061e-16, -3.366e+00,1.412e-01,
#  -1.794e+00,5.363e-01,-3.386e+00,4.852e-01,-2.021e+00,1.107e+00,
# -3.406e+00,8.379e-01,-2.023e+00,1.746e+00,-3.426e+00,1.232e+00,
# -2.010e+00,1.619e+00,-2.229e+00,1.625e+00,-1.903e+00,1.981e+00,
# -1.981e+00,2.415e+00,-2.063e+00,2.810e+00,-2.042e+00,3.171e+00,
# -1.943e+00,3.713e+00,-1.892e+00,4.619e+00,-1.913e+00,4.755e+00,
# -1.545e+00,4.862e+00,-1.167e+00,4.938e+00,-7.822e-01,4.985e+00,
#  -3.923e-01,5.000e+00,0.000e+00,4.985e+00,3.923e-01,4.866e+00,
# 7.707e-01,3.486e+00,8.368e-01,2.731e+00,8.873e-01,1.821e+00,
# 7.544e-01,1.460e+00,7.442e-01,1.429e+00,8.759e-01,1.430e+00,
# 1.039e+00,8.490e-01,7.251e-01,8.515e-01,8.515e-01,8.526e-01,
# 9.983e-01,5.121e-01,7.049e-01,5.056e-01,8.250e-01,5.065e-01,
# 9.940e-01,9.464e-01,2.285e+00,7.401e-01,2.278e+00,1.637e-01,
# 6.820e-01,1.370e-01,8.647e-01,1.777e-01,2.258e+00,1.378e-16,
#   2.251e+00]
#
# x0 = np.array(obs)
# x0[2] = 100.0
# x0[3] = 0.0
# x0[4] = 0.0
# x0[5] = 0
# x0[6] = np.pi/2.0
# x0[7] = 0.0
# x0[8] = 0.0
# Ad = A_f(x0, [0, 0, 0, 0, 0])
# Bd = B_f(x0, [0, 0, 0, 0, 0])
# print(x0)
# y = Ad.dot(x0) + Bd.dot(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
# print(y-x0)
# time.sleep(1)
# exit()

# while True:
#     inp = np.array([0.1, 0.0, 0.1, 0.1, 0.0])
#     obs, _, _, _  = env.step(inp)
#     print(proc_obs(obs))
#     time.sleep(0.02)
while True:
    lb[:nx] = -x0
    ub[:nx] = -x0
    Ad = A_f(x0, [0, 0, 0, 0, 0])
    Bd = B_f(x0, [0, 0, 0, 0, 0])
    A, lb, ub, q, P = calc_matrices(Ad, Bd, x0, x0)
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, warm_start=True)
    # prob.update(l=lb, u=ub)
    res = prob.solve()
    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    inp = res.x[-N*nu:-(N-1)*nu]
    inp[3] = -inp[3]
    obs, r, d, _ = env.step(inp)
    obs = proc_obs(obs)
    x0 = np.array(obs)
    print(inp)
    print(obs)
    time.sleep(0.02)
