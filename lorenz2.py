import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver, LstsqL2
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp
from nengolib.synapses import ss2sim
from nengolib.signal import LinearSystem, cont2discrete, s

from neuron_types import LIF, Izhikevich, Wilson, NEURON, nrnReset
from utils import LearningNode, trainDF
from plotter import plotActivities

from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.spatial import distance

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
palette = sns.color_palette('colorblind')
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif',
	rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

def measure_correlation_integral(X, l, N, seed):
	rng = np.random.RandomState(seed=seed)
	times = rng.choice(range(len(X)), size=(N,2), replace=False)
	Xi = X[times[:,0]]
	Xj = X[times[:,1]]
	delta = np.linalg.norm(Xi-Xj, axis=1)
	n_lesser = len(np.where(delta<l)[0])
	C = 1/np.square(N) * n_lesser
	return l, C

def feedback(x):
	sigma = 10
	beta = 8.0 / 3
	rho = 28
	tau = 0.1
	dx = sigma*(x[1] - x[0])
	dy = -x[1] - x[0]*x[2] + rho*x[0]
	dz = x[0]*x[1] - beta*x[2]
	return [dx, dy, dz]

def makeKick(tKick=1, seed=0):
	rng = np.random.RandomState(seed=seed)
	kick = rng.uniform(-10,10,size=3)
	stim_func = lambda t: kick if t<tKick else [0,0,0]
	return stim_func

def go(neuron_type, nEns=300, nPre=300, r=40, seed=0, t=100, tKick=1, dt=1e-3,
		max_rates=Uniform(30, 40), stim_func=lambda t: [0,0,0], fPre=DoubleExp(1e-3, 1e-2),
		learnDF=False, test=False, d=None, f=None, w=None):

	with nengo.Network(seed=seed) as model:
		inpt = nengo.Node(stim_func)  # kick
		tarX = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
		pre = nengo.Ensemble(nPre, 3, max_rates=max_rates, radius=r, neuron_type=nengo.LIF(), seed=seed)
		ens = nengo.Ensemble(nEns, 3, max_rates=max_rates, radius=r, neuron_type=neuron_type, seed=seed)
		tarA = nengo.Ensemble(nEns, 3, max_rates=max_rates, radius=r, neuron_type=nengo.LIF(), seed=seed)
		nengo.Connection(inpt, pre, synapse=None)
		nengo.Connection(inpt, tarX, synapse=None)
		# nengo.Connection(pre, ens, synapse=fPre)
		nengo.Connection(pre, tarA, synapse=fPre)
		nengo.Connection(tarX, tarX, function=feedback, synapse=~s)
		if learnDF:
			# nengo.Connection(tar, ens, synapse=None)
			nengo.Connection(tarX, pre, synapse=None)
		if test:
			# nengo.Connection(ens.neurons, ens, transform=d.T, synapse=f)        
			nengo.Connection(tarA.neurons, tarA, transform=d.T, synapse=f)        
		pEns = nengo.Probe(ens.neurons, synapse=None)
		pTarA = nengo.Probe(tarA.neurons, synapse=None)
		pTarX = nengo.Probe(tarX, synapse=None)
	with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
		sim.run(t, progress_bar=True)
	return dict(
		ens=sim.data[pEns],
		tarA=sim.data[pTarA],
		tarX=sim.data[pTarX],
		)

def train(neuron_type, nTrain, nTest, tTrain, tTest, eRate, seed=0, tBias=30,
	nEns=300, dt=1e-3, penalty=0, reg=1e-3, evals=20,
	load=[]):

	if 1 in load:
		data = np.load(f"data/lorenz_{neuron_type}.npz")
		d, rise, fall = data['d'], data['rise'], data['fall']
	else:
		targets = np.zeros((nTrain, int(tTrain/dt), 3))
		spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
		for n in range(nTrain):
			stim_func = makeKick(seed=seed+n)			
			data = go(neuron_type, learnDF=True, stim_func=stim_func,
				nEns=nEns, t=tTrain, dt=dt, seed=seed)
			targets[n] = data['tarX']  # adding filters is technically correct but breaks things
			# spikes[n] = data['ens']
			spikes[n] = data['tarA']
		d, rise, fall = trainDF(spikes, targets, nTrain, network="lorenz", neuron_type=neuron_type, ens="ens", dt=dt,
			penalty=penalty, reg=reg, evals=evals, seed=seed)
		np.savez(f"data/lorenz_{neuron_type}.npz", d=d, rise=rise, fall=fall)
		f = DoubleExp(rise, fall)
		print('synapse', rise, fall)

		tarX = targets[-1]
		aEns = f.filt(spikes[-1], dt=dt)
		xhat = np.dot(aEns, d)

		rates = np.max(aEns, axis=0)
		fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((12, 6)))
		sns.histplot(rates, stat='percent', ax=ax)
		sns.histplot(d.ravel(), stat='percent', ax=ax2)
		ax.set(title='firing rates')
		ax2.set(title='decoder')
		fig.savefig(f"plots/lorenz/lorenz_rates_{neuron_type}.pdf", bbox_inches="tight", pad_inches=0.1)

		fig = plt.figure(figsize=((18, 4)))
		ax = fig.add_subplot(111, projection='3d')
		ax2 = fig.add_subplot(122, projection='3d')
		ax.plot(*xhat.T, linewidth=0.25)
		ax2.plot(*tarX.T, linewidth=0.25)
		ax.set(title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
		ax2.set(title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
		ax.grid(False)
		ax2.grid(False)
		plt.tight_layout()
		plt.savefig(f"plots/lorenz/lorenz_decode_{neuron_type}.pdf", bbox_inches="tight", pad_inches=0.1)
	
	print('estimating error')
	for n in range(nTest):
		stim_func = makeKick(seed=100+seed+n)
		data = go(neuron_type, test=True, stim_func=stim_func,
			nEns=nEns, t=tTest, dt=dt, seed=seed,
			d=d, f=f)
		# tarX = fTarget.filt(fTarget.filt(data['tarX'], dt=dt), dt=dt)
		tarX = data['tarX']
		# aEns = f.filt(data['ens'], dt=dt)
		aEns = f.filt(data['tarA'], dt=dt)
		xhat = np.dot(aEns, d)
		fig = plt.figure(figsize=((18, 4)))
		ax = fig.add_subplot(111, projection='3d')
		ax2 = fig.add_subplot(122, projection='3d')
		ax.plot(*xhat.T, linewidth=0.25)
		ax2.plot(*tarX.T, linewidth=0.25)
		ax.set(title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
		ax2.set(title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
		ax.grid(False)
		ax2.grid(False)
		plt.tight_layout()
		fig.savefig(f'plots/lorenz/state_{neuron_type}_{n}.pdf')

		ls_tarX, Cs_tarX, ls_xhat, Cs_xhat = [], [], [], []
		for l in [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10]:
			l_tarX, C_tarX = measure_correlation_integral(tarX, l=l, N=10000, seed=seed)
			l_xhat, C_xhat = measure_correlation_integral(xhat, l=l, N=10000, seed=seed)
			if C_tarX > 0:
				ls_tarX.append(l_tarX)
				Cs_tarX.append(C_tarX)
			if C_xhat > 0:
				ls_xhat.append(l_xhat)
				Cs_xhat.append(C_xhat)
		slope_tarX, intercept_tarX, r_tarX, p_tarX, se_tarX = linregress(np.log2(ls_tarX), np.log2(Cs_tarX))
		slope_xhat, intercept_xhat, r_xhat, p_xhat, se_xhat = linregress(np.log2(ls_xhat), np.log2(Cs_xhat))
		error_tarX = slope_tarX / 2.05
		error_xhat = slope_xhat / 2.05
		fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((7,4)))
		ax.scatter(np.log2(ls_xhat), np.log2(Cs_xhat))
		ax2.scatter(np.log2(ls_tarX), np.log2(Cs_tarX))
		ax.plot(np.log2(ls_xhat), intercept_xhat+slope_xhat*np.log2(ls_xhat), label=f'slope={slope_xhat:.3}, r={r_xhat:.3}, p={p_xhat:.3f}')
		ax2.plot(np.log2(ls_tarX), intercept_tarX+slope_tarX*np.log2(ls_tarX), label=f'slope={slope_tarX:.3}, r={r_tarX:.3}, p={p_tarX:.3f}')
		ax.set(xlabel='log(l)', ylabel='log(C)', title='xhat')
		ax2.set(xlabel='log(l)', ylabel='log(C)', title='target')
		ax.legend()
		ax2.legend()
		fig.savefig(f"plots/lorenz/correlation_integrals_{neuron_type}_{n}.pdf")
		plt.close('all')

		rates = np.max(aEns, axis=0)
		fig, ax, = plt.subplots(figsize=((4, 4)))
		sns.histplot(rates, stat='percent', ax=ax)
		ax.set(title='firing rates')
		fig.savefig(f"plots/lorenz/lorenz_rates_test_{neuron_type}_{n}.pdf", bbox_inches="tight", pad_inches=0.1)

train(neuron_type=LIF(), nEns=100, tTrain=100, tTest=100, nTrain=5, nTest=5, eRate=0, load=[])