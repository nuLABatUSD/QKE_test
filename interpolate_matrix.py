import numpy as np
import matplotlib.pyplot as plt

linspace_max = 10.

x_lag, w_lag = np.polynomial.laguerre.laggauss(5)
eps = np.zeros(201+5)
eps[:201] = np.linspace(0, linspace_max, 201)
eps[201:] = x_lag + linspace_max

def dens_array(x, fun_e, fun_m):
    result = np.zeros((len(x), 4))
    for i in range(len(x)):
        fe = fun_e(x[i])
        fm = fun_m(x[i])

        P0 = (fe + fm)
        Pz = (fe - fm)/(P0 + 1e-100)

        result[i, :] = [P0, 0, 0, Pz]
    return result

def index_below(x, eps_array):
    if x > eps_array[-1]:
        return -1
    ind = np.where(eps_array <= x)
    return ind[0][-1]

def interpolate_log_linear(x, x_array, y_array):
    index = index_below(x, x_array)

    if y_array[index] * y_array[index+1] <= 0:
        return linear_fit(x, x_array[index:index+2], y_array[index:index+2])
    else:
        y_temp = linear_fit(x, x_array[index:index+2], np.log(np.abs(y_array[index:index+2])))
        if y_array[index] > 0:
            return np.exp(y_temp)
        else:
            return -np.exp(y_temp)

def interpolate_log_fifth(x, x_array, y_array):
    index = index_below(x, x_array)

    ind = max(0, index-2)
    ind = min(len(x_array)-1-4, ind)

    for i in range(1,5):
        if y_array[ind] * y_array[ind+i] <= 0:
            return fifth_order_fit(x, x_array[ind:ind+5], y_array[ind:ind+5])
    y_temp = fifth_order_fit(x, x_array[ind:ind+5], np.log(np.abs(y_array[ind:ind+5])))
    if y_array[ind] > 0:
        return np.exp(y_temp)
    else:
        return - np.exp(y_temp)

def fifth_order_fit(x, x_data, y_data):
    p = 0.0

    for j in range(5):
        Lj = 1.0
        for i in range(5):
            if i != j:
                Lj *= (x - x_data[i])/(x_data[j] - x_data[i])
        p += y_data[j] * Lj
    return p

def linear_fit(x, x_data, y_data):
    z_fit = np.polyfit(x_data, y_data, 1)
    p_fit = np.poly1d(z_fit)
    return p_fit(x)

def interpolate_matrix(x, x_array, P_arrays, fun=interpolate_log_linear):
    rho_ee = np.zeros(len(x_array))
    rho_mm = np.zeros_like(rho_ee)

    rho_ee[:] = 0.5 * P_arrays[:,0] * (1 + P_arrays[:,3])
    rho_mm[:] = 0.5 * P_arrays[:,0] * (1 - P_arrays[:,3])

    fe_temp = fun(x, x_array, rho_ee)
    fm_temp = fun(x, x_array, rho_mm)

    return np.array([fe_temp + fm_temp, 0, 0, (fe_temp - fm_temp)/(fe_temp + fm_temp + 1e-100)])

def do_interpolation(x, x_array, P_arrays, fun=interpolate_log_linear):
    rho_ee = np.zeros(len(x_array))
    rho_mm = np.zeros_like(rho_ee)

    rho_ee[:] = 0.5 * P_arrays[:,0] * (1 + P_arrays[:,3])
    rho_mm[:] = 0.5 * P_arrays[:,0] * (1 - P_arrays[:,3])

    if x < linspace_max:
        func = interpolate_log_fifth
    else:
        func = interpolate_log_linear

    fe_temp = func(x, x_array, rho_ee)
    fm_temp = func(x, x_array, rho_mm)

    return np.array([fe_temp + fm_temp, 0, 0, (fe_temp - fm_temp)/(fe_temp + fm_temp + 1e-100)])


def diff_plots(eps, rho, fun_e, fun_m, fun=interpolate_log_linear, fun_comp=interpolate_matrix):
    eps_interp = np.linspace(0.01, np.max(eps)-0.01, 10000)
    interp_rho_P = np.zeros((len(eps_interp), 4))
    interp_rho_mat = np.zeros_like(interp_rho_P)
    exact_P = np.zeros_like(interp_rho_P)
    
    for i in range(len(eps_interp)):
        for j in range(4):
            interp_rho_P[i,j] = fun(eps_interp[i], eps, rho[:,j])
            interp_rho_mat[i,:] = fun_comp(eps_interp[i], eps, rho, fun)
        fe = fun_e(eps_interp[i])
        fm = fun_m(eps_interp[i])
        P0 = (fe+fm)
        Pz = (fe-fm)/(P0+1e-100)
        exact_P[i, :] = [P0, 0, 0, Pz]

    Delta0_P = np.abs(interp_rho_P[:,0]-exact_P[:,0])/exact_P[:,0]
    Deltaz_P = np.abs(interp_rho_P[:,3]-exact_P[:,3])/np.abs(exact_P[:,3])

    Delta0_mat = np.abs(interp_rho_mat[:,0]-exact_P[:,0])/exact_P[:,0]
    Deltaz_mat = np.abs(interp_rho_mat[:,3]-exact_P[:,3])/np.abs(exact_P[:,3])

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,4))
    ax[0].semilogy(eps_interp, Delta0_P, label='fitting P')
    ax[0].semilogy(eps_interp, Delta0_mat, label='fitting matrix')
    ax[0].legend(loc='upper right')
    ax[1].semilogy(eps_interp, Deltaz_P)
    ax[1].semilogy(eps_interp, Deltaz_mat)
