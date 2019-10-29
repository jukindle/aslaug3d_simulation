import osqp
import numpy as np
import scipy.sparse as sps
import sympy as sp

N = 25

rotmat = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a)], [sp.sin(a), sp.cos(a)]])

vars = sp.symbols('x_0:97')
inps = sp.symbols('u_0:5')
l1, l2, l3 = sp.Matrix([[1.0], [0.0]]), sp.Matrix([[1.0], [0.0]]), sp.Matrix([[0.1], [0.0]])
r_t_ee = rotmat(vars[5]) * (l1 + rotmat(vars[6]) * (l2 + l3))
J = r_t_ee.jacobian(vars[5:7])
r_v_ee = J*sp.Matrix(vars[7:9])
r_v_wee = sp.Matrix(vars[2:4]) + sp.Matrix([[-vars[4], 0.0], [0.0, vars[4]]])*r_v_ee
ee_v_w = -rotmat(vars[5] + vars[6])*r_v_wee

sp_n = sp.Matrix(vars[0:2]) - ee_v_w*0.02  # NOTE: or plus?
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
# state_new_f = sp.lambdify((vars, inps), state_new, 'numpy')  # state_new((1,...,97), (1,...,5)) returns shape (97, 1)
A_f = sp.lambdify((vars, inps), state_new.jacobian(vars), 'numpy')
B_f = sp.lambdify((vars, inps), state_new.jacobian(inps), 'numpy')
A = A_f(np.zeros(97), np.zeros(5))
B = B_f(np.zeros(97), np.zeros(5))


idx_ins = (N+1)*97
n_vars = (N+1)*(97+5)
n_cons_eq = N*97

E_dyn =



#
# states = sp.symbols("s_0:{}".format(n_vars))
# cons = sp.zeros(n_cons_eq, 1)
# for i in range(N):
#     states_i = states[(i*97):((i+1)*97)]
#     ins_i = states[(idx_ins+i*5):(idx_ins+(i+1)*5)]
#     states_ip1 = states[((i+1)*97):((i+2)*97)]
#     cons[(i*97):((i+1)*97), 0] = sp.Matrix(states_ip1) - state_new_fsp(states_i, ins_i)
#
# cons_f = sp.lambdify((states,), cons, 'numpy')
#
# cost = states[N*97]**2 + states[N*98]**2
# cost_f = sp.lambdify((states,), cost, 'numpy')
# # cost_jac_f = sp.lambdify((states,), cost.jacobian(states), 'numpy')
