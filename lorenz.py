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

# def feedback(x):
# 	sigma = 10
# 	beta = 8.0 / 3
# 	rho = 28
# 	tau = 0.1
# 	dx = sigma*(x[1] - x[0])
# 	dy = -x[1] - x[0]*x[2] + rho*x[0]
# 	dz = x[0]*x[1] - beta*x[2]
# 	return [dx, dy, dz]

def feedback(x):
	sigma = 10
	beta = 8.0 / 3
	rho = 28
	tau = 0.1
	dx = sigma*(x[1] - x[0])
	dy = -x[0]*x[2] - x[1]  
	dz = x[0]*x[1] - beta*(x[2] + rho) - rho
	return [dx/3, dy/3, dz/3]

def makeKick(tKick=1e-2, seed=0):
	rng = np.random.RandomState(seed=seed)
	kick = rng.uniform(-1,1,size=3)
	stim_func = lambda t: kick if t<=tKick else [0,0,0]
	return stim_func

def kickAndReset(seed=0):
	rng = np.random.RandomState(seed=seed)
	kicks = []
	for T in range(100):
		kicks.append(rng.uniform(-10,10,size=3))
	stim_func = lambda t: kicks[int(t/10)] if (t%10)<1 else [0,0,0]
	return stim_func

def go(neuron_type, nEns=300, nPre=300, seed=0, t=100, dt=1e-3,
		max_rates=Uniform(30, 40), intercepts=Uniform(-0.8, 0.8), radius=40,
		stim_func=lambda t: [0,0,0], noise_drive=False,
		fPre=DoubleExp(1e-3, 1e-2), fSmooth=DoubleExp(1e-2, 1e-1), fBias=DoubleExp(1e-3, 1e-1),
		eRate=0, dRate=3e-6,
		learnB=False, learnFF=False, learnDF=False, learnFB=False, test=False,
		fFB=None, dB=None, eB=None, wB=None, dFF=None, eFF=None, wFF=None, dFB=None, eFB=None, wFB=None):

	weightsB = wB if (np.any(wB) and not learnB) else np.zeros((nPre, nEns))
	weightsFF = wFF if (np.any(wFF) and not learnFF) else np.zeros((nPre, nEns))
	weightsFB = wFB if (np.any(wFB) and not learnFF and not learnFB) else np.zeros((nEns, nEns))
	rng = np.random.RandomState(seed=seed)
	with nengo.Network(seed=seed) as model:
		inpt = nengo.Node(stim_func)
		noise = nengo.Node(lambda t: rng.normal(2.0, size=3) if noise_drive else [0,0,0])
		const = nengo.Node([1,1,1])
		tarX = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
		bias = nengo.Ensemble(nPre, 3, max_rates=max_rates, seed=seed)
		pre = nengo.Ensemble(nPre, 3, max_rates=max_rates, radius=radius, seed=seed)
		ens = nengo.Ensemble(nEns, 3, max_rates=max_rates, radius=radius, neuron_type=neuron_type, seed=seed)
		tarA = nengo.Ensemble(nEns, 3, max_rates=max_rates, intercepts=intercepts, radius=radius, seed=seed)
		nengo.Connection(inpt, pre, synapse=None)
		nengo.Connection(noise, pre, synapse=None)
		nengo.Connection(inpt, tarX, synapse=None)
		nengo.Connection(tarX, tarX, function=feedback, synapse=~s)
		cB = nengo.Connection(bias, ens, synapse=fBias, solver=NoSolver(weightsB, weights=True), seed=seed)
		cFF = nengo.Connection(pre, ens, synapse=fPre, solver=NoSolver(weightsFF, weights=True), seed=seed)
		if learnB:  # learn a bias connection that feeds spikes representing zero to all ensembles
			nB = LearningNode(bias, ens, 3, conn=cB, d=dB, e=eB, w=wB, eRate=eRate, dRate=dRate)
			nengo.Connection(bias.neurons, nB[:nPre], synapse=fBias)
			nengo.Connection(ens.neurons, nB[nPre: nPre+nEns], synapse=fSmooth)
			nengo.Connection(tarA.neurons, nB[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
			nengo.Connection(const, nB[-3:], synapse=None)
			nengo.Connection(nB, ens.neurons, synapse=None)     
		if learnFF:  # learn to receive supervised "recurrent" input from ReLU
			nengo.Connection(tarX, pre, synapse=None, seed=seed)
			nengo.Connection(tarX, tarA, synapse=fPre, seed=seed)
			nFF = LearningNode(pre, ens, 3, conn=cFF, d=dFF, e=eFF, w=wFF, eRate=eRate, dRate=dRate)
			nengo.Connection(pre.neurons, nFF[:nPre], synapse=fPre)
			nengo.Connection(ens.neurons, nFF[nPre: nPre+nEns], synapse=fSmooth)
			nengo.Connection(tarA.neurons, nFF[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
			nengo.Connection(tarX, nFF[-3:], synapse=None)
			nengo.Connection(nFF, ens.neurons, synapse=None)
		if learnDF:
			nengo.Connection(tarX, pre, synapse=None)
			# nengo.Connection(pre, tarA, synapse=fPre)
		if learnFB: # learn to receive supervised "recurrent" input from ens
			tarX2 = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
			ens2 = nengo.Ensemble(nEns, 3, max_rates=max_rates, radius=radius, neuron_type=neuron_type, seed=seed)
			tarA2 = nengo.Ensemble(nEns, 3, max_rates=max_rates, intercepts=intercepts, radius=radius, seed=seed)
			nengo.Connection(tarX, pre, synapse=None, seed=seed)
			nengo.Connection(tarX, tarX2, synapse=fPre, seed=seed)
			nengo.Connection(tarX2, tarA2, synapse=fFB, seed=seed)
			# nengo.Connection(tarX2, tarA2, synapse=None, seed=seed)
			cB2 = nengo.Connection(bias, ens2, synapse=fBias, solver=NoSolver(weightsB, weights=True), seed=seed)
			cFF2 = nengo.Connection(ens, ens2, synapse=fFB, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
			nFF2 = LearningNode(ens, ens2, 3, conn=cFF2, d=dFB, e=eFB, w=wFB, eRate=eRate, dRate=0)
			nengo.Connection(ens.neurons, nFF2[:nEns], synapse=fFB)
			nengo.Connection(ens2.neurons, nFF2[nEns: 2*nEns], synapse=fSmooth)
			nengo.Connection(tarA2.neurons, nFF2[2*nEns: 3*nEns], synapse=fSmooth)
			nengo.Connection(nFF2, ens2.neurons, synapse=None)
		if test:
			cFB = nengo.Connection(ens, ens, synapse=fFB, solver=NoSolver(weightsFB, weights=True))  # recurrent
			# nengo.Connection(tarX, pre, synapse=None)
			# nengo.Connection(tarA.neurons, tarA, transform=dFB.T, synapse=fFB)        
		pEns = nengo.Probe(ens.neurons, synapse=None)
		pTarA = nengo.Probe(tarA.neurons, synapse=None)
		pTarX = nengo.Probe(tarX, synapse=None)
		pEns2 = nengo.Probe(ens2.neurons, synapse=None) if learnFB else None
		pTarA2 = nengo.Probe(tarA2.neurons, synapse=None) if learnFB else None
	with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
		sim.run(t, progress_bar=True)
	if learnB:
		dB, eB, wB = nB.d, nB.e, nB.w  
	if learnFF:
		dFF, eFF, wFF = nFF.d, nFF.e, nFF.w  
	if learnFB:
		dFB, eFB, wFB = nFF2.d, nFF2.e, nFF2.w  
	return dict(
		times=sim.trange(),
		ens=sim.data[pEns],
		tarA=sim.data[pTarA],
		tarX=sim.data[pTarX],
		ens2=sim.data[pEns2] if learnFB else None,
		tarA2=sim.data[pTarA2] if learnFB else None,
		dB=dB, eB=eB, wB=wB,
		dFF=dFF, eFF=eFF, wFF=wFF,
		dFB=dFB, eFB=eFB, wFB=wFB,
		)

def train(neuron_type, nTrain, nTest, tTrain, tTest, eRate, seed=0, tTrans=0, noise_tar=False, noise_drive=False,
	nEns=100, dt=1e-3, penalty=0, reg=1e-2, evals=20, fSmooth=DoubleExp(1e-2, 1e-1),
	load=[]):

	rng = np.random.RandomState(seed=seed)

	if 0 in load:
		data = np.load(f"data/lorenz_{neuron_type}.npz")
		dB, eB, wB = data['dB'], data['eB'], data['wB']
	else:
		print('train dB, eB, wB from bias to ens')
		dB, eB, wB = None, None, None
		for n in range(nTrain):
			data = go(neuron_type, learnB=True, eRate=10*eRate, noise_drive=False,
				nEns=nEns, t=20, dt=dt, seed=seed,
				dB=dB, eB=eB, wB=wB)
			dB, eB, wB = data['dB'], data['eB'], data['wB']
			np.savez(f"data/lorenz_{neuron_type}.npz",
				dB=dB, eB=eB, wB=wB)
			plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
				"lorenz", neuron_type, "bias", n, nTrain)

	if 1 in load:
		data = np.load(f"data/lorenz_{neuron_type}.npz")
		dFF, eFF, wFF = data['dFF'], data['eFF'], data['wFF']
	else:
		print('train dFF, eFF, wFF from pre to ens')
		dFF, eFF, wFF = None, None, None
		for n in range(nTrain):
			stim_func = makeKick(seed=seed+n)
			data = go(neuron_type, learnFF=True, eRate=eRate, stim_func=stim_func, noise_drive=noise_drive,
				nEns=nEns, t=tTrain, dt=dt, seed=seed,
				wB=wB,
				dFF=dFF, eFF=eFF, wFF=wFF)
			dFF, eFF, wFF = data['dFF'], data['eFF'], data['wFF']
			np.savez(f"data/lorenz_{neuron_type}.npz",
				dB=dB, eB=eB, wB=wB,
				dFF=dFF, eFF=eFF, wFF=wFF)
			plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
				"lorenz", neuron_type, "ens", n, nTrain)

	if 2 in load:
		data = np.load(f"data/lorenz_{neuron_type}.npz")
		dFB, rise, fall = data['dFB'], data['rise'], data['fall']
		fFB = DoubleExp(rise, fall)
	else:
		print('train decoders and filters')
		fPre = DoubleExp(1e-3, 1e-2)
		targets = np.zeros((2*nTrain, int((tTrain-tTrans)/dt), 3))
		spikes = np.zeros((2*nTrain, int((tTrain-tTrans)/dt), nEns))
		for n in range(2*nTrain):
			stim_func = makeKick(seed=seed+n)			
			data = go(neuron_type, learnDF=True, stim_func=stim_func, noise_drive=noise_drive,
				nEns=nEns, t=tTrain, dt=dt, seed=seed,
				wB=wB,
				wFF=wFF)
			times = data['times']
			tarX = data['tarX']
			if noise_tar:
				# tarX = tarX + np.array([0.5*np.sin(20*np.pi*times), 0.4*np.sin(18*np.pi*times), 0.3*np.sin(22*np.pi*times)]).T
				tarX = tarX + rng.normal(0.2, size=((times.shape[0], 3)))
			# targets[n] = data['tarX'][int(tTrans/dt):]
			targets[n] = fPre.filt(tarX, dt=dt)[int(tTrans/dt):]
			spikes[n] = data['ens'][int(tTrans/dt):]
			# spikes[n] = data['tarA'][int(tTrans/dt):]
		dFB, rise, fall = trainDF(spikes, targets, 2*nTrain, network="lorenz", neuron_type=neuron_type, ens="ens", dt=dt,
			penalty=penalty, reg=reg, evals=evals, seed=seed)
		np.savez(f"data/lorenz_{neuron_type}.npz",
			dB=dB, eB=eB, wB=wB,
			dFF=dFF, eFF=eFF, wFF=wFF,
			dFB=dFB, rise=rise, fall=fall)
		fFB = DoubleExp(rise, fall)
		print('synapse', rise, fall)

		for n in range(2*nTrain):
			tarX = targets[n][int(tTrans/dt):]
			aEns = fFB.filt(spikes[n], dt=dt)
			xhat = np.dot(aEns, dFB)[int(tTrans/dt):]

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
			fig.savefig(f"plots/lorenz/lorenz_decode_{neuron_type}_{n}.pdf", bbox_inches="tight", pad_inches=0.1)
			plt.close('all')

	if 3 in load:
		data = np.load(f"data/lorenz_{neuron_type}.npz")
		eFB, wFB = data['eFB'], data['wFB']
	else:
		print('train eFB, wFB from ens to ens2')
		eFB, wFB = None, None
		for n in range(nTrain):
			stim_func = makeKick(seed=seed+n)
			data = go(neuron_type, learnFB=True, eRate=eRate/2, stim_func=stim_func, noise_drive=noise_drive,
				nEns=nEns, t=tTrain, dt=dt, seed=seed,
				wB=wB,
				wFF=wFF,
				dFB=dFB, fFB=fFB,
				eFB=eFB, wFB=wFB)
			eFB, wFB = data['eFB'], data['wFB']
			np.savez(f"data/lorenz_{neuron_type}.npz",
				dB=dB, eB=eB, wB=wB,
				dFF=dFF, eFF=eFF, wFF=wFF,
				dFB=dFB, rise=rise, fall=fall,
				eFB=eFB, wFB=wFB)
			plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
				"lorenz", neuron_type, "ens2", n, nTrain)
		times = data['times']
		tar = data['tarX']
		tarX = tar[int(tTrans/dt):]
		aEns = fFB.filt(data['ens2'], dt=dt)
		xhat = np.dot(aEns, dFB)[int(tTrans/dt):]
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
		fig.savefig(f'plots/lorenz/check_ens2.pdf')
	
	print('estimating error')
	for n in range(nTest):
		stim_func = makeKick(seed=100+seed+n)
		# stim_func = kickAndReset(seed=100+seed+n)
		data = go(neuron_type, test=True, stim_func=stim_func, noise_drive=False,
			nEns=nEns, t=tTest, dt=dt, seed=seed,
			dFB=dFB, wB=wB, wFF=wFF, wFB=wFB, fFB=fFB)
		times = data['times']
		tar = data['tarX']
		tarX = tar[int(tTrans/dt):]
		aEns = fFB.filt(data['ens'], dt=dt)
		xhat = np.dot(aEns, dFB)[int(tTrans/dt):]
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

		rates = np.max(aEns, axis=0)
		fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((12, 6)))
		sns.histplot(rates, stat='percent', ax=ax)
		sns.histplot(dFB.ravel(), stat='percent', ax=ax2)
		ax.set(title='firing rates')
		ax2.set(title='decoder')
		fig.savefig(f"plots/lorenz/rates_{neuron_type}_{n}.pdf")
		plt.close('all')

train(neuron_type=LIF(), nEns=100, tTrain=100, tTest=100, nTrain=5, nTest=3, tTrans=0, eRate=1e-8, noise_drive=True, load=[0])