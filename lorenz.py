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
from multiprocessing import Pool, current_process
from pdb import set_trace
from misc import nanarray
from numpy.linalg import norm

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')
plt.rc('axes', titlesize='xx-large')


M = 40
dt = 0.002 # we use Euler forward
n_ga = 2
alpha = 5
# n_thread = 1
n_thread = os.cpu_count()-1

# subscript management
I = np.arange(M)
I1 = np.roll(I, -1) # [1,2,...,0]
Im1 = np.roll(I, 1) # [-1,0,...,]
Im2 = np.roll(I, 2) # [-2,-1,...,]


def F(x, ga):
    F = x[Im1]*(x[I1]-x[Im2]) - x[I] + ga[0] - 0.01*x**2
    return F

def nabla_F(x):
    Fx = np.zeros([M,M]) # the first axis labels components of f
    Fx[I,Im1] = x[I1]-x[Im2]
    Fx[I,I1] = x[Im1]
    Fx[I,Im2] = -x[Im1]
    Fx[I,I] = -1 - 0.02*x
    return Fx

def delta_F(x):
    dF = nanarray([n_ga, M])
    dF[0] = 1
    dF[1] = 0
    return dF

def sig(x, ga):
    sig = np.exp(- x@x / 2)
    return sig + ga[1]

def nabla_sig(x):
    sig = np.exp(- x@x / 2)
    sigx = - sig * x
    return sigx

def delta_sig(x):
    dsig = nanarray([n_ga])
    dsig[0] = 0
    dsig[1] = 1
    return dsig

# def get_phi(x): return x.mean()
# def nabla_phi(x): return np.ones(M)/M

def get_phi(x): return (x**2).mean()
def nabla_phi(x): return 2*x/M

# def get_phi(x): return np.sin(x.sum())
# def nabla_phi(x): return np.cos(x.sum()) * np.ones(M)


def path_ker(T, W, ga):
    np.random.seed()
    N_T = int(T/dt)
    N_W = int(W/dt)
   
    # forward process
    x, db = nanarray([2, N_T+N_W+1, M])
    phi = nanarray([N_T+N_W+1])
    x[0] = np.random.normal(size=M, scale = 2) 
    phi[0] = get_phi(x[0])
    for n in range(int(5/dt)):
        _ = np.random.normal(size=M, scale=dt**0.5)
        x[0] = x[0] + F(x[0], ga) * dt + sig(x[0], ga) * _
    for n in range(N_T+N_W):
        db[n] = np.random.normal(size=M, scale=dt**0.5)
        x[n+1] = x[n] + F(x[n], ga) * dt + sig(x[n], ga) * db[n]
        phi[n+1] = get_phi(x[n+1])
    phiavg = phi.mean()
    phi -= phiavg

    # backpropagation
    nu = nanarray([N_T+1, M])
    nu[-1] = 0
    for k in range(N_T-1, -1, -1):
        nu[k] = (1 - alpha*dt) * nu[k+1] \
                + nabla_F(x[k]).T @ nu[k+1] * dt \
                + nabla_sig(x[k]) * (db[k] @ nu[k+1]) \
                + nabla_phi(x[k]) * dt \
                + alpha / sig(x[k], ga) * phi[k+1:k+N_W+1].sum() * db[k] * dt

    # linear response
    grad = nanarray([N_T, n_ga])
    for k in range(N_T):
        grad[k] = delta_F(x[k]) @ nu[k+1] + db[k] @ nu[k+1] * delta_sig(x[k]) / dt
    # grad1, grad2 = nanarray([2, N_T, n_ga])
    # for k in range(N_T):
        # grad1[k] = delta_F(x[k]) @ nu[k+1] 
        # grad2[k] = db[k] @ nu[k+1] * delta_sig(x[k]) / dt
        
    # fig = plt.figure(figsize=(10,8))
    # plt.plot(grad2, label='grad2')
    # plt.plot(grad1, label='grad1')
    # plt.xlabel('$n$')
    # plt.ylabel('$grad1 and 2$')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('grad_time_{}.png'.format(alpha))
    # plt.close()
    # set_trace()
    return phiavg, grad[N_W:-N_W].mean(0) 


def get_phiavg_no_noise(T, ga):
    np.random.seed()
    N_T = int(T/dt)
    x = nanarray([N_T+1, M])
    phi = nanarray([N_T+1])
    x[0] = np.random.normal(size=M, scale = 2) 
    for n in range(int(5/dt)):
        x[0] = x[0] + F(x[0], ga) * dt 

    phi[0] = get_phi(x[0])
    for n in range(N_T):
        x[n+1] = x[n] + F(x[n], ga) * dt
        phi[n+1] = get_phi(x[n+1])
    return phi.mean()


def draw(T, W, N):
    ga_0 = np.linspace(6, 10, N)
    ga_1 = np.linspace(2, 6, N)
    ga = np.array(np.meshgrid(ga_0, ga_1)).transpose(1,2,0)
    
    try:
        ga, grad, phi, phiavg_no_noise = pickle.load( open("contour.p", "rb"))
    except FileNotFoundError:
        grad = nanarray([N, N, n_ga])
        phi = nanarray([N, N])
        phiavg_no_noise = nanarray(N)
        for i in range(N):
            for j in range(N):
                print(i,j)
                phi[i,j], grad[i,j] = path_ker(T, W, ga[i,j])
        for j in range(N):
            print('no noise, ', j)
            phiavg_no_noise[j] = get_phiavg_no_noise(T, ga[0,j])
        pickle.dump((ga, grad, phi, phiavg_no_noise), open("contour.p", "wb"))

    # contour
    fig = plt.figure(figsize=(7,6))
    ax = plt.axes()
    CS = plt.contourf(ga[...,0], ga[...,1], phi, 20, cmap=plt.cm.bone, origin='lower')
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('$\Phi_{avg}$')
    plt.xlabel('$\\gamma^0$')
    plt.ylabel('$\\gamma^1$')
   
    plt.scatter(ga[...,0], ga[...,1], color='k', s=15)
    Q = plt.quiver(ga[...,0], ga[...,1], grad[...,0], grad[...,1], units='x',
            pivot='tail', width=0.02, color='r', scale=10)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('lorenz_contour.png')
    plt.close()

    # line
    stepsize = (ga_0[-1]-ga_0[0]) / N
    plt.figure(figsize=[7,6])
    for i in range(N):
        plt.plot(ga_0, phi[i], marker='o', linestyle='-', markersize=6)
        A = stepsize / 2 
        for x, y, slope in zip(ga_0, phi[i], grad[i,:,0]):
            AA = A / ((1+slope**2)**0.2)
            plt.plot([x-AA, x+AA], [y-slope*AA, y+slope*AA], color='black', linestyle='-')
    plt.plot(ga_0, phiavg_no_noise, marker='^', linestyle='None', color='black', markersize=12, label='no noise')

    plt.ylabel('$\Phi^{avg}$')
    plt.xlabel('$\gamma^0$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("lorenz_lines_ga0.png")
    plt.close()

    stepsize = (ga_1[-1]-ga_1[0]) / N
    plt.figure(figsize=[7,6])
    for i in range(N):
        plt.plot(ga_1, phi[:,i], marker='o', linestyle='-', markersize=6)
        A = stepsize / 2 # short line length in the plot
        for x, y, slope in zip(ga_1, phi[:,i], grad[:,i,1]):
            AA = A / ((1+slope**2)**0.2)
            plt.plot([x-AA, x+AA], [y-slope*AA, y+slope*AA], color='black', linestyle='-')
    plt.ylabel('$\Phi^{avg}$')
    plt.xlabel('$\gamma^1$')
    plt.tight_layout()
    plt.savefig("lorenz_lines_ga1.png")
    plt.close()


def orbit(T, ga):
    np.random.seed()
    N_T = int(T/dt)
    x = nanarray([N_T+1, M])
    x[0] = np.random.normal(size=M, scale = 2) 
    for n in range(int(5/dt)):
        db = np.random.normal(size=M, scale=dt**0.5)
        x[0] = x[0] + F(x[0], ga) * dt + sig(x[0], ga) * db
    for n in range(N_T):
        db = np.random.normal(size=M, scale=dt**0.5)
        x[n+1] = x[n] + F(x[n], ga) * dt + sig(x[n], ga) * db
    fig = plt.figure(figsize=(7,6))
    plt.plot(x[:,0], x[:,1], 'k-')
    plt.ylabel('$x^1$')
    plt.xlabel('$x^0$')
    plt.tight_layout()
    plt.savefig('lorenz_orbit.png')
    plt.close()
  

orbit(2, [8,2])
draw(2000,2,6)
