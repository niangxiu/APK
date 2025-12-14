# lorenz example with automatic differentiation
from __future__ import division
import shutil
import sys
import os
import time
import pickle
import itertools
import numpy as np
from numpy import newaxis, sin, cos, pi, tanh, cosh, sqrt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool, current_process
from pdb import set_trace
from misc import nanarray, npnanarray
from numpy.linalg import norm
import jax
import jax.numpy as jnp

# output to a log file
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
log_file = open("log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)  # optional: also log errors

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')
plt.rc('axes', titlesize='xx-large')
# np.set_printoptions(formatter={'float': '{:2.1e}'.format})
np.set_printoptions(formatter={'float': '{:.1f}'.format})


L = 10
M = 10 # dimension of the system
N = 8 # dimension of the observable
dt = 0.002
T = 5.
N_T = int(T/dt)
alpha = 3
C = 0.05

I = np.arange(M)
I1 = np.roll(I, -1) # [1,2,...,0]
Im1 = np.roll(I, 1) # [-1,0,...,]
Im2 = np.roll(I, 2) # [-2,-1,...,]
zeros = lambda z: jax.tree_map(jnp.zeros_like, z)


@jax.jit
def get_F(x, gaF):
    F = x[Im1]*(x[I1]-x[Im2]) - x[I] + gaF
    return F

@jax.jit
def nabla_F(x, gaF):
    return jax.jacobian(get_F, argnums=0)(x, gaF)

@jax.jit
def nabla_ga_F(x, gaF):
    return jax.jacobian(get_F, argnums=1)(x, gaF)


@jax.jit
def get_xi(x, gaXi, xstar):
    return gaXi * ((xstar-x) @ (xstar-x)) * (xstar-x)

@jax.jit
def nabla_xi(x, gaXi, xstar):
    return jax.jacobian(get_xi, argnums=0)(x, gaXi, xstar)

@jax.jit
def nabla_ga_xi(x, gaXi, xstar):
    return jax.jacobian(get_xi, argnums=1)(x, gaXi, xstar)

@jax.jit
def nabla_star_xi(x, gaXi, xstar):
    return jax.jacobian(get_xi, argnums=2)(x, gaXi, xstar)


@jax.jit
def get_phi(x):
    # return jnp.array(x[0::2])
    # return jnp.array(x[0:2])
    return x[jnp.r_[0:4, 5:9]]

@jax.jit
def nabla_phi(x):
    return jax.jacobian(get_phi)(x)


@jax.jit
def forward(x0, gaF, sig, gaXi, xstar, y, key):
    x   = nanarray((N_T + 1, M))     # path
    db  = nanarray((N_T, M))         # Brownian increments
    xi  = nanarray((N_T, M))         # control
    phi = nanarray((N_T, N))         # observations/features
    x = x.at[0].set(x0)

    def body_fun(n, carry):
        x, db, xi, phi, key = carry

        x_n = x[n]
        phi_n = get_phi(x_n)                     # shape (N,)
        xi_n  = get_xi(x_n, gaXi, xstar[n])      # shape (M,)
        key, subkey = jax.random.split(key)
        db_n = jax.random.normal(subkey, (M,)) * jnp.sqrt(dt)
        F_n = get_F(x_n, gaF)                    # shape (M,)

        x_next = x_n + F_n * dt + sig * db_n + xi_n * dt

        x   = x.at[n + 1].set(x_next)
        db  = db.at[n].set(db_n)
        xi  = xi.at[n].set(xi_n)
        phi = phi.at[n].set(phi_n)
        return (x, db, xi, phi, key)

    x, db, xi, phi, key = jax.lax.fori_loop(
        0, N_T, body_fun, (x, db, xi, phi, key))

    Phi = (((phi - y) ** 2).sum() + C * (xi ** 2).sum()) * dt / 2.0 / T
    return x, db, xi, phi, Phi


def draw_path(phi, i_num, y_compare=None):
    fig, ax = plt.subplots(figsize=[6,6])
    plt.plot(phi[:,1], phi[:,2], 'k-')
    if y_compare is not None:
        plt.plot(y_compare[:,1], y_compare[:,2], 'r-')
    plt.ylabel('$x^2$')
    plt.xlabel('$x^1$')
    plt.savefig("assimi_orbit_T={}_{}.png".format(T, i_num), dpi=150, bbox_inches="tight")
    plt.close()
    return


# generate data
gaF_true = 8.0
sig_true = 0.0
x0_true = np.array([-6.9,-0.5,1.5,9.3,0.9,1.3,0.2,2.6,6.7,2.7])
key = jax.random.PRNGKey(200)
_, _, _, y, _ = forward(x0_true, gaF_true, sig_true, 0, np.zeros([N_T, M]), np.zeros([N_T, N]), key)
draw_path(y, 0)
# x_true, db, xi, y, Phi = forward(x0_true, gaF_true, sig_true, 0, np.zeros([N_T, M]), np.zeros([N_T, N]), key)
# print('x_true[-1] = ', x_true[-1])


@jax.jit
def apk(x0, gaF, sig, gaXi, xstar, Phi_avg, alpha, y, key):
    x, db, xi, phi, Phi = forward(x0, gaF, sig, gaXi, xstar, y, key)
    Phit = Phi - Phi_avg

    # backward pass
    nu = jnp.full((N_T + 1, M), jnp.nan)
    nu = nu.at[-1].set(0.0)
    def backward_body(i, nu):
        k = N_T - 1 - i
        nu_k1 = nu[k + 1]
        nu_k = (1.0 - alpha * dt) * nu_k1
        nu_k = nu_k + (nabla_F(x[k], gaF).T @ nu_k1) * dt
        nu_k = nu_k + (nabla_xi(x[k], gaXi, xstar[k]).T @ nu_k1) * dt
        nu_k = nu_k + (nabla_phi(x[k]).T @ (phi[k] - y[k])) * dt
        nu_k = nu_k + (alpha / sig) * Phit * db[k]
        nu_k = nu_k + C * (nabla_xi(x[k], gaXi, xstar[k]).T @ xi[k]) * dt
        nu = nu.at[k].set(nu_k)
        return nu

    nu = jax.lax.fori_loop(0, N_T, backward_body, nu)

    grad_x0 = nu[0] / T
    grad_gaF = jnp.zeros_like(gaF)
    grad_sig = jnp.array(0.0)
    grad_gaXi = jnp.zeros_like(gaXi)
    grad_xstar = jnp.full((N_T, M), jnp.nan)

    def grad_body(k, carry):
        grad_gaF, grad_sig, grad_gaXi, grad_xstar = carry
        nu_k1 = nu[k + 1]
        tmp = nu_k1 + C * xi[k]
        grad_gaF = grad_gaF + (nabla_ga_F(x[k], gaF).T @ nu_k1) * dt / T
        grad_sig = grad_sig + (db[k] @ nu_k1) / T
        grad_gaXi = grad_gaXi + (nabla_ga_xi(x[k], gaXi, xstar[k]).T @ tmp) * dt / T
        grad_xstar_k = (nabla_star_xi(x[k], gaXi, xstar[k]).T @ tmp) * dt / T
        grad_xstar = grad_xstar.at[k].set(grad_xstar_k)
        return (grad_gaF, grad_sig, grad_gaXi, grad_xstar)

    grad_gaF, grad_sig, grad_gaXi, grad_xstar = jax.lax.fori_loop(
        0, N_T, grad_body, (grad_gaF, grad_sig, grad_gaXi, grad_xstar)
    )

    return grad_x0, grad_gaF, grad_sig, grad_gaXi, grad_xstar, Phi


def get_delta(prime, grad, eta, Phi_avg, lower_cap=False, factor=1):
    C1 = np.abs(Phi_avg) / 10 / np.linalg.norm(grad)**2
    C2 = np.clip(eta, -C1, C1)
    v = -C2 * grad
    if lower_cap is not False:
        v = max(v, lower_cap-prime)
    return v


def assimilate():
    key = jax.random.PRNGKey(10)
    N_optimize = 2001

    # resume or initial guess
    try:
        x0, gaF, sig, gaXi, xstar, Phi_avgs = pickle.load(open("parameters.p", "rb"))
        Phi_avg_min = min(Phi_avgs)
    except FileNotFoundError:
        # x0 = np.random.normal(size=[M,], scale = 5) 
        x0 = np.zeros([M,]) 
        gaF = 13.0
        sig = 2.0
        gaXi = 0.1
        xstar = np.zeros([N_T, M])
        Phi_avg_min = 5000
        Phi_avgs = npnanarray([N_optimize])

    for i_optimize in range(N_optimize):
        base_key, key = jax.random.split(key)
        keys = jax.random.split(base_key, L)   # shape (L, 2)

        # compute Phi_avg, draw one sample path, report status
        def forward_with_key(subkey):
            _, _, _, phi_l, Phi_l = forward(x0, gaF, sig, gaXi, xstar, y, subkey)
            return phi_l, Phi_l   # scalar
        phi, Phi = jax.vmap(forward_with_key)(keys)   # shape (L,)
        Phi_avgs[i_optimize] = Phi_avg = jnp.mean(Phi)

        if Phi_avg < Phi_avg_min * 0.95: 
            Phi_avg_min = Phi_avg
            draw_path(phi.mean(0), i_optimize+1, y)
        print('round {}: {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(i_optimize, Phi_avg, gaF, sig, gaXi))
        print(x0, xstar[:1])
        # set_trace()

        # copute gradients
        def apk_with_key(subkey):
            return apk(x0, gaF, sig, gaXi, xstar, Phi_avg, alpha, y, subkey)
        grad_x0, grad_gaF, grad_sig, grad_gaXi, grad_xstar, Phi = jax.vmap(apk_with_key)(keys)

        # update x0, gaF, sig, gaXi, xstar, Phi_avg
        eta, eta_x0 = 1., 1.
        if Phi_avg <= 6: 
            eta, eta_x0 = 0.1, 0.1*alpha
        # if Phi_avg <= 1: 
            # eta, eta_x0 = 0.01, 0.01*alpha
        delta_x0    = get_delta(x0, grad_x0.mean(0), eta_x0, Phi_avg)
        delta_gaF   = get_delta(gaF,   grad_gaF.mean(0),   eta, Phi_avg)
        delta_sig   = get_delta(sig,   grad_sig.mean(0),   eta, Phi_avg, 0.5)
        delta_gaXi  = get_delta(gaXi,  grad_gaXi.mean(0),  eta, Phi_avg, 0.1)
        delta_xstar = get_delta(xstar, grad_xstar.mean(0), eta/dt, Phi_avg)
        x0 += delta_x0
        gaF += delta_gaF
        sig += delta_sig
        gaXi += delta_gaXi
        xstar += delta_xstar

        print('projected contribution: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
                (delta_x0*grad_x0.mean(0)).sum(), 
                delta_gaF*grad_gaF.mean(0), 
                delta_sig*grad_sig.mean(0), 
                delta_gaXi*grad_gaXi.mean(0), 
                (delta_xstar*grad_xstar.mean(0)).sum()))
    
    pickle.dump((x0, gaF, sig, gaXi, xstar, Phi_avgs), open("parameters.p", "wb"))
    return Phi_avgs


starttime = time.time()
# Phi_avgs = assimilate()
endtime = time.time()
print('time spent (seconds):', endtime-starttime)


fig, ax = plt.subplots(figsize=[6,5])
_, _, _, _, _, Phi_avgs = pickle.load(open("parameters.p", "rb"))
plt.semilogy(Phi_avgs, 'k-', label='L=10')
plt.xlabel('# epoch')
plt.ylabel('$\\Phi^{avg}$')
plt.savefig('Lorenz96_Phi_hist_T={}.png'.format(T), dpi=150, bbox_inches="tight")
plt.close()

x0, gaF, sig, gaXi, xstar, Phi_avgs = pickle.load(open("parameters.p", "rb"))
x, db, xi, phi, Phi = forward(x0, gaF, 0, gaXi, xstar, y, key)
print(Phi)
draw_path(phi, 9999, y)
