import numpy as np
import sympy as sp
import sys
import quadprog
sys.path.append("../..")
from envs.aslaug_v1_cont import AslaugEnv

# Define params
N = 10
l1, l2, l3 = sp.Matrix([[0.2], [0.0]]), sp.Matrix([[0.2], [0.0]]), sp.Matrix([[0.1], [0.0]])

# Function for calculating rotation matrix
rotmat = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a)], [sp.sin(a), sp.cos(a)]])

# Calculate dynamics of system for single step
vars = sp.symbols('x_0:97')
inps = sp.symbols('u_0:5')
r_t_ee = rotmat(vars[5]) * (l1 + rotmat(vars[6]) * (l2 + l3))
J = r_t_ee.jacobian(vars[5:7])
r_v_ee = J*sp.Matrix(vars[7:9])
r_v_wee = sp.Matrix(vars[2:4]) + sp.Matrix([[-vars[4], 0.0], [0.0, vars[4]]])*r_v_ee
ee_v_w = -rotmat(vars[5] + vars[6])*r_v_wee

sp_n = sp.Matrix(vars[0:2]) + ee_v_w*0.02  # NOTE: or plus?
vmb_n = sp.Matrix(vars[2:5]) + sp.Matrix(inps[0:3])
j_n = sp.Matrix(vars[5:7]) + sp.Matrix(vars[7:9])*0.02
jv_n = sp.Matrix(vars[7:9]) + sp.Matrix(inps[3:5])

r_l1_p = rotmat(vars[5])*l1/2.0
r_l2_p = rotmat(vars[5])*(l1 + rotmat(vars[6])*l2/2.0)

ls_n = sp.Matrix((r_l1_p, vars[5:6], r_l2_p, vars[6:7]))

rotmat_huge_diag = [rotmat(0.02*vars[4]).T for x in range(41)]
shifts_huge = sp.Matrix([0.02*vars[2+(x % 2)] for x in range(2*41)])
sc_n = sp.diag(*rotmat_huge_diag)*(sp.Matrix(vars[15:97]) - shifts_huge)

state_new = sp.Matrix([sp_n, vmb_n, j_n, jv_n, ls_n, sc_n])
state_new_fsp = sp.lambdify((vars, inps), state_new, 'sympy')  # NOTE: seems to be ok, checked.

# Calculate dynamics of system for concatenated steps
idx_ins = (N+1)*97
n_vars = (N+1)*97 + N*5
n_cons_eq_dyn = N*97

states = sp.symbols("s_0:{}".format(n_vars))
print("Setting up constraints")
cons_dyn = sp.zeros(n_cons_eq_dyn, 1)
# cons_lidar = sp.zeros((N+1)*41, 1)
cons_link1 = sp.zeros((N+1)*41, 1)
cons_link2 = sp.zeros((N+1)*41, 1)

lid_sel = sp.diag(*([[1, 1]]*41)).T

for i in range(N):
    states_i = states[(i*97):((i+1)*97)]
    ins_i = states[(idx_ins+i*5):(idx_ins+(i+1)*5)]
    states_ip1 = states[((i+1)*97):((i+2)*97)]
    cons_dyn[(i*97):((i+1)*97), 0] = sp.Matrix(states_ip1) - state_new_fsp(states_i, ins_i)

    # Lidar to base
    lid_sq_i = sp.Matrix(states_i[15:97]).applyfunc(lambda x: x**2)
    # cons_lidar[(i*41):((i+1)*41), 0] = lid_sel*lid_sq_i

    # Lidar to links
    lnk1, lnk2 = states_i[9:12], states_i[12:15]
    lidlnk1 = sp.Matrix(states_i[15:97]) - sp.Matrix(lnk1[0:2]*41)
    lidlnk2 = sp.Matrix(states_i[15:97]) - sp.Matrix(lnk2[0:2]*41)
    lidlnk1_sq = lidlnk1.applyfunc(lambda x: x**2)
    lidlnk2_sq = lidlnk2.applyfunc(lambda x: x**2)
    cons_link1[(i*41):((i+1)*41), 0] = lid_sel*lidlnk1_sq
    cons_link2[(i*41):((i+1)*41), 0] = lid_sel*lidlnk2_sq

cons = sp.Matrix([cons_dyn[:, 0]])
#cons = sp.Matrix([cons_dyn[:, 0], cons_link1[:, 0], cons_link2[:, 0]])
print("Setting up jacobian of constraints")
A_eq_f = sp.lambdify((states,), cons.jacobian(states), 'numpy')
print("DONE SETTING UP STUFF")

# Setup env
env = AslaugEnv(gui=True)
obs = env.reset()

# Define QP solver
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T) + np.eye(P.shape[0])*1e-4  # make sure P is symmetric
    qp_a = -q
    if A is not None:
        if G is None:
            qp_C = -A.T
            qp_b = -b
        else:
            qp_C = -np.vstack([A, G]).T
            qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

# Define function for converting observation to state
def proc_obs(obs):
    pos_sp = np.array([obs[2], -obs[1]])
    mb_vel = obs[6:9]
    linkpos = [0.5, 0, 0, 1.0, 0, 0]
    j_pos = obs[45:47]
    j_vel = obs[47:49]
    sc = obs[49:90]
    angs = np.linspace(-np.pi/2, np.pi/2, 41)
    sc_xs, sc_ys = np.cos(angs)*sc, np.sin(angs)*sc
    scf = np.vstack((sc_xs, sc_ys)).T.reshape((-1,))
    obs_proc = np.concatenate((pos_sp, mb_vel, j_pos, j_vel, linkpos, scf)).tolist()
    return obs_proc


obs_proc = proc_obs(obs)
x0 = obs_proc*(N+1) + N*[0.03, 0.0, 0.0, 0.0, 0.0]

lb = (N)*([-1e20, -1e20, -0.4, -0.4, -0.75, -2.89, -1.76, -1.0, -1.0, -1e20, -1e20, -1e20, -1e20, -1e20, -1e20] + 82*[-1e20]) + N*[-0.03, -0.03, -0.01, -0.015, -0.015]
ub = (N)*([+1e20, +1e20, +0.4, +0.4, +0.75, +2.89, +1.76, +1.0, +1.0, +1e20, +1e20, +1e20, +1e20, +1e20, +1e20] + 82*[1e20]) + N*[+0.03, +0.03, +0.01, +0.015, +0.015]
lb[9:15] = [-1e20, -1e20, -1e20, -1e20, -1e20, -1e20]
ub[9:15] = [+1e20, +1e20, +1e20, +1e20, +1e20, +1e20]
cl = n_cons_eq_dyn*[0.0]# + (N+1)*41*[-1e20] + (N+1)*41*[-1e20]
cu = n_cons_eq_dyn*[0.0]# + (N+1)*41*[1e20] + (N+1)*41*[1e20]


A_inp_fix = np.hstack((np.eye(97), np.zeros((97, len(x0)-97))))

A_eq_dyn = A_eq_f(x0)
A_eq = np.vstack((A_eq_dyn, A_inp_fix))
b = np.concatenate((np.zeros(A_eq_dyn.shape[0]), obs_proc))
P = np.zeros((len(x0), len(x0)))
P[N*97:N*97+2, N*97:N*97+2] = np.eye(2)*100.0
q = np.zeros(len(x0))

# Setup state ineq constraints
n_leq = len(lb)
A_ineq_states_leq = np.hstack((np.zeros((n_leq, 97)), np.eye(len(ub))))
b_ineq_states_leq = np.array(ub)

A_ineq_states_geq = np.hstack((np.zeros((n_leq, 97)), -np.eye(len(lb))))
b_ineq_states_geq = -np.array(lb)

A_in = np.vstack((A_ineq_states_leq, A_ineq_states_geq))
b_in = np.concatenate((b_ineq_states_leq, b_ineq_states_geq))


x = quadprog_solve_qp(P, q, A=A_eq, b=b, G=A_in, h=b_in)
inp = np.array(x[idx_ins:(idx_ins+5)])

while True:
    obs, r, d, _ = env.step(inp)
    obs_proc = proc_obs(obs)
    x[:(97)] = obs_proc
    #x0 = obs_proc*(N+1) + N*[0.0, 0.0, 0.0, 0.0, 0.0]

    A_eq_dyn = A_eq_f(x)
    A_eq = np.vstack((A_eq_dyn, A_inp_fix))
    b = np.concatenate((np.zeros(A_eq_dyn.shape[0]), obs_proc))
    x = quadprog_solve_qp(P, q, A=A_eq, b=b, G=A_in, h=b_in)

    inp = np.array(x[idx_ins:(idx_ins+5)])
    print(inp)
