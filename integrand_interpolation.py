import numpy as np
import matplotlib.pyplot as plt

import interpolate_matrix as int

eta_e = 0.01
eta_m = -0.01
def FD(x, n):
    return 1/(np.exp(x-n)+1)
def f_FD_e(x):
    return FD(x, eta_e)
def f_FD_m(x):
    return FD(x, eta_m)

def Fvv_eq(p):
    f = np.zeros(5)
    g = np.zeros(5)
    for i in range(1,5):
        f[i] = f_FD_e(p[i-1])
        g[i] = f_FD_m(p[i-1])
        
    return Fvv(f, g)
    
def make_p_test(eps):
    eps_len = len(eps)
    p1_vals = np.concatenate((eps[10:201:10], eps[201:]))
    p2_vals = np.concatenate((eps[5:201:10], eps[201:]))
    val_len = 0
    for i in range(len(p1_vals)):
        for j in range(len(p2_vals)):
            for k in range(eps_len):
                if p1_vals[i] + p2_vals[j] >= eps[k] and p1_vals[i] + p2_vals[j] - eps[k] <= np.max(eps):
                    val_len += 1
    p_test = np.zeros((val_len, 4))
    
    p_index = 0
    for i in range(len(p1_vals)):
        for j in range(len(p2_vals)):
            for k in range(eps_len):
                if p1_vals[i] + p2_vals[j] >= eps[k] and p1_vals[i] + p2_vals[j] - eps[k] <= np.max(eps):
                    p_test[p_index, :] = [p1_vals[i], p2_vals[j], eps[k], p1_vals[i] + p2_vals[j] - eps[k]]
                    p_index += 1
    return p_test
    
##################
# For the function Fvv(f, g), choose f[1] = f(p1), the nu_e distribution. To keep this notation, f has length 5, and f[0] is ignored, with f[1], f[2], f[3], f[4]. And likewise for g and the nu_mu distribution
#
def Fvv(f, g):
    T1 = 2*f[3]*f[4]*(1-f[2])*(1-f[1])
    T1 += 2*g[3]*g[4]*(1-g[2])*(1-g[1])
    T1 += f[3]*g[4]*(1-g[2])*(1-f[1])
    T1 += g[3]*f[4]*(1-f[2])*(1-g[1])
    
    T2 = 2*(1-f[3])*(1-f[4])*f[2]*f[1]
    T2 += 2*(1-g[3])*(1-g[4])*g[2]*g[1]
    T2 += (1-f[3])*(1-g[4])*g[2]*f[1]
    T2 += (1-g[3])*(1-f[4])*f[2]*g[1]

    #T1 = 2*f3*f4*(1-f2)*(1-f1) + f3*g4*(1-g2)*(1-f1) + 2*g3*g4*(1-g2)*(1-g1) + g3*f4*(1-f2)*(1-g1)
    #T2 = 2*(1-f3)*(1-f4)*f2*f1 + (1-f3)*(1-g4)*g2*f1 + 2*(1-g3)*(1-g4)*g2*g1 + (1-g3)*(1-f4)*f2*g1
    return T1, T2   

def convert_density_to_fg(density):
    return 0.5 * density[0] * (1 + density[3]), 0.5 * density[0] * (1 - density[3])

def make_fg_interpolate(p_arr, eps_array, dens_arrays):
    ind_list = []
    for i in range(4):
        ind_temp = np.where(eps_array == p_arr[i])[0]
        if len(ind_temp) == 1:
            ind_list.append(ind_temp[0])
        else:
            ind_list.append(len(eps_array))
        
    results = np.zeros((2,5))
    
    for i in range(4):
        if ind_list[i] < len(eps_array):
            results[0,i+1], results[1,i+1] = convert_density_to_fg(dens_arrays[ind_list[i]])
        else:
            results[0,i+1], results[1,i+1] = convert_density_to_fg(int.do_interpolation(p_arr[i], eps_array, dens_arrays))
            
    return results

def J1(p1, p2, p3):
    return 16./15 * p3**3 * (10 * (p1+p2)**2 - 15 * (p1+p2) * p3 + 6 * p3**2)

def J2(p1, p2):
    return 16./15 * p2**3 * (10 * p1**2 + 5 * p1*p2 + p2**2)
    
def J3(p1, p2, p3):
    return 16./15 * ((p1+p2)**5 - 10 * (p1+p2)**2 * p3**3 + 15 * (p1+p2) * p3**4 - 6 * p3**5)

def J(p1, p2, p3):
    if p2 <= p1:
        if p3 <= p2:
            return J1(p1, p2, p3)
        if p3 <= p1:
            return J2(p1, p2)
        if p3 <= p1+p2:
            return J3(p1, p2, p3)
        else:
            return 0
    else:
        if p3 <= p2:
            return J1(p1, p2, p3)
        if p3 <= p1:
            return J2(p2, p1)
        if p3 <= p1+p2:
            return J3(p1, p2, p3)
        else:
            return 0