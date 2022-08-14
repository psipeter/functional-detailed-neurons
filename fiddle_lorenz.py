import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from nengolib.signal import s
from utils import trainDF
from nengolib import Lowpass, DoubleExp
from nengo.solvers import NoSolver, LstsqL2
from nengo.dists import Uniform

def measure_correlation_integral(X, l, N, seed):
	rng = np.random.RandomState(seed=seed)
	times = rng.choice(range(len(X)), size=(N,2), replace=False)
	Xi = X[times[:,0]]
	Xj = X[times[:,1]]
	delta = np.linalg.norm(Xi-Xj, axis=1)
	n_lesser = len(np.where(delta<l)[0])
	C = 1/np.square(N) * n_lesser
	return l, C

def feedback_tar(x):
	speed = 0.25
	sigma = 10
	beta = 8.0 / 3
	rho = 28
	tau = 0.1
	dx = sigma*(x[1] - x[0])
	dy = -x[0]*x[2] - x[1]  
	dz = x[0]*x[1] - beta*(x[2] + rho) - 40
	deriv = np.array([dx, dy, dz])
	return speed * deriv

def make_kick(tKick=1e-3, seed=0):
	rng = np.random.RandomState(seed=seed)
	kick = rng.uniform(-1,1,size=3)
	kick_func = lambda t: kick if t<=tKick else [0,0,0]
	return kick_func

def make_noise(t, dt=0.001, freq=1, rms=1, seed=0):
    n1 = nengo.processes.WhiteSignal(period=t, high=freq, rms=rms, seed=seed)
    n2 = nengo.processes.WhiteSignal(period=t, high=freq, rms=rms, seed=200+seed)
    n3 = nengo.processes.WhiteSignal(period=t, high=freq, rms=rms, seed=400+seed)
    with nengo.Network() as model:
        i1 = nengo.Node(n1)
        i2 = nengo.Node(n2)
        i3 = nengo.Node(n3)
        inpt = nengo.Node(size_in=3)
        nengo.Connection(i1, inpt[0], synapse=None)
        nengo.Connection(i2, inpt[1], synapse=None)
        nengo.Connection(i3, inpt[2], synapse=None)
        probe = nengo.Probe(inpt, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = sim.data[probe]
    noise_func = lambda t: u[int(t/dt)]
    return noise_func

def run(
		nEns = 300,
		nPre = 300,
		r = 30,
		neuron_type = nengo.LIF(),
		seed = 0,
		t = 100,
		tTrans = 0,
		tKick = 1e-3,
		nTrain = 1,
		nTest = 1,
		evals = 10,
		dt = 1e-3,
		reg = 1e-2,
		penalty = 0,
		max_rates = Uniform(30, 40),
		intercepts = Uniform(-1, 1),
		load = [],
		spk_thr = 100,
		fPre = Lowpass(1e-2),
		noise_freq = 1.0,
		noise_rms = 1.0,
	):

	rng = np.random.RandomState(seed=seed)

	def go(phase, kick_func, noise_func, d, f):
		with nengo.Network(seed=seed) as model:
			kick = nengo.Node(kick_func)
			noise = nengo.Node(noise_func)
			tar = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
			pre = nengo.Ensemble(nPre, 3, max_rates=max_rates, radius=r, neuron_type=nengo.LIF(), seed=seed)
			ens = nengo.Ensemble(nEns, 3, max_rates=max_rates, radius=r, intercepts=intercepts, neuron_type=neuron_type, seed=seed)
			nengo.Connection(kick, tar, synapse=None)
			nengo.Connection(noise, tar, synapse=None)
			nengo.Connection(tar, tar, function=feedback_tar, synapse=~s)
			if phase=="train":
				nengo.Connection(tar, pre, synapse=None)
				nengo.Connection(pre, ens, synapse=fPre)
			elif phase=="test":
				nengo.Connection(kick, pre, synapse=None)
				nengo.Connection(pre, ens, synapse=fPre)
				nengo.Connection(ens.neurons, ens, transform=d.T, synapse=f)        
			spk = nengo.Probe(ens.neurons, synapse=None)
			ptar = nengo.Probe(tar, synapse=None)
		with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
			sim.run(t, progress_bar=True)
		return sim.trange(), sim.data[spk], sim.data[ptar]

	if 1 in load:
		data = np.load(f"data/lorenz_fiddle.npz")
		d, rise, fall = data['d'], data['rise'], data['fall']
		f = DoubleExp(rise, fall)
	else:
		targets = np.zeros((nTrain, int((t-tTrans)/dt), 3))
		spikes = np.zeros((nTrain, int((t-tTrans)/dt), nEns))
		for n in range(nTrain):
			# print(f"decoder and filter simulations, iteration {n}")
			kick_func = make_kick(seed=seed+n)
			noise_func = make_noise(t=t, freq=noise_freq, rms=noise_rms, seed=seed+n)
			times, spk, tar = go("train", kick_func, noise_func, None, None)
			# pretend that neurons below spike threshold did not spike at all
			spk_cut = np.zeros_like(spk).T
			for nrn in range(nEns):
				if np.sum(spk[:,nrn]*dt) < spk_thr and n==nTrain-1:
					print(f"removing quiet neuron {nrn} with {np.sum(spk[:,nrn]*dt)} spikes")
					pass
				else:
					spk_cut[nrn] = spk[:,nrn].T
			targets[n] = tar[int(tTrans/dt):]
			# spikes[n] = spk[int(tTrans/dt):]
			spikes[n] = spk_cut.T[int(tTrans/dt):]
		d, rise, fall = trainDF(spikes, targets, nTrain, network="lorenz", neuron_type="LIF", ens="ens", dt=dt,
			penalty=penalty, seed=seed, reg=reg, evals=evals)
		print('synapse', rise, fall)
		# print("d", d)
		np.savez(f"data/lorenz_fiddle.npz", d=d, rise=rise, fall=fall)

		f = DoubleExp(rise, fall)
		for n in range(nTrain):
			aEns = f.filt(spikes[n], dt=dt)[int(tTrans/dt):]
			xhat = np.dot(aEns, d)
			target = targets[n][int(tTrans/dt):]
			fig, axes = plt.subplots(nrows=2, ncols=3, figsize=((9, 6)), sharex=False, sharey=False)
			axes[0][0].plot(xhat[:,0], xhat[:,1], linewidth=0.25)
			axes[0][1].plot(xhat[:,0], xhat[:,2], linewidth=0.25)
			axes[0][2].plot(xhat[:,1], xhat[:,2], linewidth=0.25)
			axes[1][0].plot(target[:,0], target[:,1], linewidth=0.25)
			axes[1][1].plot(target[:,0], target[:,2], linewidth=0.25)
			axes[1][2].plot(target[:,1], target[:,2], linewidth=0.25)
			axes[0][0].set(xlim=((-40, 40)), ylim=((-40, 40)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$")
			axes[0][1].set(xlim=((-40, 40)), ylim=((-40, 40)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{z}$")
			axes[0][2].set(xlim=((-40, 40)), ylim=((-40, 40)), title='xhat', xlabel=r"$\mathbf{y}$", ylabel=r"$\mathbf{z}$")
			axes[1][0].set(xlim=((-40, 40)), ylim=((-40, 40)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$")
			axes[1][1].set(xlim=((-40, 40)), ylim=((-40, 40)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{z}$")
			axes[1][2].set(xlim=((-40, 40)), ylim=((-40, 40)), title='target', xlabel=r"$\mathbf{y}$", ylabel=r"$\mathbf{z}$")
			plt.tight_layout()
			fig.savefig(f"plots/lorenz/fiddle_decode_{n}.pdf")

	f = DoubleExp(rise, fall)

	for n in range(nTest):
		kick_func = make_kick(seed=seed+n)
		noise_func = lambda t: [0,0,0]
		# noise_func = make_noise(t=t, freq=noise_freq, rms=noise_rms, seed=seed+n)
		times, spk, tar = go("test", kick_func, noise_func, d, f)
		aEns = f.filt(spk, dt=dt)[int(tTrans/dt):]
		xhat = np.dot(aEns, d)
		target = tar[int(tTrans/dt):]

		fig, axes = plt.subplots(nrows=2, ncols=3, figsize=((9, 6)), sharex=False, sharey=False)
		axes[0][0].plot(xhat[:,0], xhat[:,1], linewidth=0.25)
		axes[0][1].plot(xhat[:,0], xhat[:,2], linewidth=0.25)
		axes[0][2].plot(xhat[:,1], xhat[:,2], linewidth=0.25)
		axes[1][0].plot(target[:,0], target[:,1], linewidth=0.25)
		axes[1][1].plot(target[:,0], target[:,2], linewidth=0.25)
		axes[1][2].plot(target[:,1], target[:,2], linewidth=0.25)
		axes[0][0].set(xlim=((-40, 40)), ylim=((-40, 40)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$")
		axes[0][1].set(xlim=((-40, 40)), ylim=((-40, 40)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{z}$")
		axes[0][2].set(xlim=((-40, 40)), ylim=((-40, 40)), title='xhat', xlabel=r"$\mathbf{y}$", ylabel=r"$\mathbf{z}$")
		axes[1][0].set(xlim=((-40, 40)), ylim=((-40, 40)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$")
		axes[1][1].set(xlim=((-40, 40)), ylim=((-40, 40)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{z}$")
		axes[1][2].set(xlim=((-40, 40)), ylim=((-40, 40)), title='target', xlabel=r"$\mathbf{y}$", ylabel=r"$\mathbf{z}$")
		plt.tight_layout()
		fig.savefig(f"plots/lorenz/fiddle_state_{n}.pdf")

		aMin = np.around(np.min(aEns, axis=0), decimals=3)
		aMax = np.max(aEns, axis=0)
		fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((12, 6)))
		sns.histplot(aMin, stat='percent', ax=ax)
		sns.histplot(aMax, stat='percent', ax=ax2)
		ax.set(title='min firing rates')
		ax2.set(title='max firing rates')
		fig.savefig(f"plots/lorenz/fiddle_rates_{n}.pdf")

		ls_tarX, Cs_tarX, ls_xhat, Cs_xhat = [], [], [], []
		for l in [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10]:
			l_tarX, C_tarX = measure_correlation_integral(target[2000:], l=l, N=10000, seed=seed)
			l_xhat, C_xhat = measure_correlation_integral(xhat[2000:], l=l, N=10000, seed=seed)
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
		fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((12,6)), sharey=True)
		ax.scatter(np.log2(ls_xhat), np.log2(Cs_xhat))
		ax2.scatter(np.log2(ls_tarX), np.log2(Cs_tarX))
		ax.plot(np.log2(ls_xhat), intercept_xhat+slope_xhat*np.log2(ls_xhat), label=f'slope={slope_xhat:.3}, r={r_xhat:.3}, p={p_xhat:.3f}')
		ax2.plot(np.log2(ls_tarX), intercept_tarX+slope_tarX*np.log2(ls_tarX), label=f'slope={slope_tarX:.3}, r={r_tarX:.3}, p={p_tarX:.3f}')
		ax.set(xlabel='log(l)', ylabel='log(C)', title='xhat')
		ax2.set(xlabel='log(l)', ylabel='log(C)', title='target')
		ax.legend()
		ax2.legend()
		fig.savefig(f"plots/lorenz/fiddle_correlation_integrals_{n}.pdf")


run(nEns=100, t=200, nTrain=3, nTest=3, r=40, intercepts=Uniform(-1, 1), spk_thr=100, noise_freq=20.0, noise_rms=0.2,
	reg=1e-1, evals=20, load=[])