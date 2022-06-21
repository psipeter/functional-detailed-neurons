import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from nengolib.signal import s
from utils import trainDF
from nengolib import Lowpass, DoubleExp
from nengo.solvers import NoSolver, LstsqL2

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
	speed = 1
	sigma = 10
	beta = 8.0 / 3
	rho = 28
	tau = 0.1
	dx = sigma*(x[1] - x[0])
	dy = -x[0]*x[2] - x[1]  
	dz = x[0]*x[1] - beta*(x[2] + rho) - 40
	deriv = np.array([dx, dy, dz])
	return speed * deriv

def makeKick(tKick=1e-3, seed=0):
	rng = np.random.RandomState(seed=seed)
	# sampler = nengo.dists.UniformHypersphere()
	# kick = sampler.sample(1, 3, rng=rng).T.reshape(-1)
	kick = rng.uniform(-1,1,size=3)
	stim = lambda t: kick if t<=tKick else [0,0,0]
	return stim

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
		evals = 10,
		dt = 1e-3,
		reg = 1e-2,
		penalty = 0,
		max_rates = nengo.dists.Uniform(30, 40),
		sim_pre = True,
		load = [],
		noise_tar = False,
		noise_drive = False,
		intercept = 1.0,
		nTest = 1,
	):

	rng = np.random.RandomState(seed=seed)
	fPre = Lowpass(1e-2) if sim_pre else None

	def go(phase, kick, d, f):
		with nengo.Network(seed=seed) as model:
			inpt = nengo.Node(kick)
			noise = nengo.Node(lambda t: rng.normal(1.0, size=3) if noise_drive else [0,0,0])
			tar = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
			if sim_pre:
				pre = nengo.Ensemble(nPre, 3, max_rates=max_rates, radius=r, neuron_type=nengo.LIF(), seed=seed)
			else:
				pre = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
			ens = nengo.Ensemble(nEns, 3, max_rates=max_rates, radius=r, intercepts=nengo.dists.Uniform(-intercept, intercept), neuron_type=neuron_type, seed=seed)
			nengo.Connection(inpt, tar, synapse=None)
			nengo.Connection(tar, tar, function=feedback_tar, synapse=~s)
			if phase=="train":
				nengo.Connection(tar, pre, synapse=None)
				nengo.Connection(noise, pre, synapse=None)	
				nengo.Connection(pre, ens, synapse=fPre)
			elif phase=="test":
				nengo.Connection(inpt, pre, synapse=None)
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
			kick = makeKick(tKick=tKick, seed=n)
			times, spk, tar = go("train", kick, None, None)
			if noise_tar:
				# tar = tar + np.array([0.5*np.sin(20*np.pi*times), 0.4*np.sin(18*np.pi*times), 0.3*np.sin(22*np.pi*times)]).T
				tar = tar + rng.normal(0.2, size=((times.shape[0], 3)))
			if sim_pre:
				# targets[n] = fPre.filt(tar, dt=dt)[int(tTrans/dt):]
				targets[n] = tar[int(tTrans/dt):]
			else:
				targets[n] = tar[int(tTrans/dt):]
			spikes[n] = spk[int(tTrans/dt):]
		d, rise, fall = trainDF(spikes, targets, nTrain, network="lorenz", neuron_type="LIF", ens="ens", dt=dt,
			penalty=penalty, seed=seed, reg=reg, evals=evals)
		print('synapse', rise, fall)
		np.savez(f"data/lorenz_fiddle.npz", d=d, rise=rise, fall=fall)

		f = DoubleExp(rise, fall)
		for n in range(nTrain):
			aEns = f.filt(spikes[n], dt=dt)[int(tTrans/dt):]
			xhat = np.dot(aEns, d)
			target = targets[n][int(tTrans/dt):]
			# fig = plt.figure(figsize=((12, 6)))
			# ax = fig.add_subplot(121, projection='3d')
			# ax2 = fig.add_subplot(122, projection='3d')
			# ax.plot(*xhat.T, linewidth=0.25)
			# ax2.plot(*target.T, linewidth=0.25)
			# ax.set(title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
			# ax2.set(title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
			# ax.grid(False)
			# ax2.grid(False)
			# plt.tight_layout()
			# fig.savefig(f"plots/lorenz/fiddle_decode_{n}.pdf")
			fig, axes = plt.subplots(nrows=2, ncols=3, figsize=((9, 6)), sharex=False, sharey=False)
			axes[0][0].plot(xhat[:,0], xhat[:,1], linewidth=0.25)
			axes[0][1].plot(xhat[:,0], xhat[:,2], linewidth=0.25)
			axes[0][2].plot(xhat[:,1], xhat[:,2], linewidth=0.25)
			axes[1][0].plot(target[:,0], target[:,1], linewidth=0.25)
			axes[1][1].plot(target[:,0], target[:,2], linewidth=0.25)
			axes[1][2].plot(target[:,1], target[:,2], linewidth=0.25)
			axes[0][0].set(xlim=((-r, r)), ylim=((-r, r)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$")
			axes[0][1].set(xlim=((-r, r)), ylim=((-r, r)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{z}$")
			axes[0][2].set(xlim=((-r, r)), ylim=((-r, r)), title='xhat', xlabel=r"$\mathbf{y}$", ylabel=r"$\mathbf{z}$")
			axes[1][0].set(xlim=((-r, r)), ylim=((-r, r)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$")
			axes[1][1].set(xlim=((-r, r)), ylim=((-r, r)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{z}$")
			axes[1][2].set(xlim=((-r, r)), ylim=((-r, r)), title='target', xlabel=r"$\mathbf{y}$", ylabel=r"$\mathbf{z}$")
			plt.tight_layout()
			fig.savefig(f"plots/lorenz/fiddle_decode_{n}.pdf")

	f = DoubleExp(rise, fall)

	for n in range(nTest):
		kick = makeKick(tKick=tKick, seed=100+n)
		times, spk, tar = go("test", kick, d, f)
		aEns = f.filt(spk, dt=dt)[int(tTrans/dt):]
		xhat = np.dot(aEns, d)
		target = tar[int(tTrans/dt):]
			
		# palette = sns.color_palette("colorblind")
		# fig = plt.figure(figsize=((12, 6)))
		# ax = fig.add_subplot(121, projection='3d')
		# ax2 = fig.add_subplot(122, projection='3d')
		# ax.scatter(*xhat[0].T, color=palette[1])
		# ax2.scatter(*target[0].T, color=palette[1])
		# ax.plot(*xhat[int(tKick/dt):].T, color=palette[0], linewidth=0.25)
		# ax2.plot(*target[int(tKick/dt):].T, color=palette[0], linewidth=0.25)
		# ax.plot(*xhat[:int(tKick/dt)].T, color=palette[1], linewidth=0.25)
		# ax2.plot(*target[:int(tKick/dt)].T, color=palette[1], linewidth=0.25)
		# ax.set(xlim=((-30, 30)), ylim=((-30, 30)), zlim=((-30, 30)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
		# ax2.set(xlim=((-30, 30)), ylim=((-30, 30)), zlim=((-30, 30)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
		# ax.grid(False)
		# ax2.grid(False)
		# plt.tight_layout()
		# fig.savefig(f"plots/lorenz/fiddle_3d_{n}.pdf")

		fig, axes = plt.subplots(nrows=2, ncols=3, figsize=((9, 6)), sharex=False, sharey=False)
		axes[0][0].plot(xhat[:,0], xhat[:,1], linewidth=0.25)
		axes[0][1].plot(xhat[:,0], xhat[:,2], linewidth=0.25)
		axes[0][2].plot(xhat[:,1], xhat[:,2], linewidth=0.25)
		axes[1][0].plot(target[:,0], target[:,1], linewidth=0.25)
		axes[1][1].plot(target[:,0], target[:,2], linewidth=0.25)
		axes[1][2].plot(target[:,1], target[:,2], linewidth=0.25)
		axes[0][0].set(xlim=((-r, r)), ylim=((-r, r)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$")
		axes[0][1].set(xlim=((-r, r)), ylim=((-r, r)), title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{z}$")
		axes[0][2].set(xlim=((-r, r)), ylim=((-r, r)), title='xhat', xlabel=r"$\mathbf{y}$", ylabel=r"$\mathbf{z}$")
		axes[1][0].set(xlim=((-r, r)), ylim=((-r, r)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$")
		axes[1][1].set(xlim=((-r, r)), ylim=((-r, r)), title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{z}$")
		axes[1][2].set(xlim=((-r, r)), ylim=((-r, r)), title='target', xlabel=r"$\mathbf{y}$", ylabel=r"$\mathbf{z}$")
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

		# ls_tarX, Cs_tarX, ls_xhat, Cs_xhat = [], [], [], []
		# for l in [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10]:
		# 	l_tarX, C_tarX = measure_correlation_integral(target[2000:], l=l, N=10000, seed=seed)
		# 	l_xhat, C_xhat = measure_correlation_integral(xhat[2000:], l=l, N=10000, seed=seed)
		# 	if C_tarX > 0:
		# 		ls_tarX.append(l_tarX)
		# 		Cs_tarX.append(C_tarX)
		# 	if C_xhat > 0:
		# 		ls_xhat.append(l_xhat)
		# 		Cs_xhat.append(C_xhat)
		# slope_tarX, intercept_tarX, r_tarX, p_tarX, se_tarX = linregress(np.log2(ls_tarX), np.log2(Cs_tarX))
		# slope_xhat, intercept_xhat, r_xhat, p_xhat, se_xhat = linregress(np.log2(ls_xhat), np.log2(Cs_xhat))
		# error_tarX = slope_tarX / 2.05
		# error_xhat = slope_xhat / 2.05
		# fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((12,6)), sharey=True)
		# ax.scatter(np.log2(ls_xhat), np.log2(Cs_xhat))
		# ax2.scatter(np.log2(ls_tarX), np.log2(Cs_tarX))
		# ax.plot(np.log2(ls_xhat), intercept_xhat+slope_xhat*np.log2(ls_xhat), label=f'slope={slope_xhat:.3}, r={r_xhat:.3}, p={p_xhat:.3f}')
		# ax2.plot(np.log2(ls_tarX), intercept_tarX+slope_tarX*np.log2(ls_tarX), label=f'slope={slope_tarX:.3}, r={r_tarX:.3}, p={p_tarX:.3f}')
		# ax.set(xlabel='log(l)', ylabel='log(C)', title='xhat')
		# ax2.set(xlabel='log(l)', ylabel='log(C)', title='target')
		# ax.legend()
		# ax2.legend()
		# fig.savefig(f"plots/lorenz/fiddle_correlation_integrals_{n}.pdf")



run(nEns=100, t=200, nTrain=5, nTest=5, r=35, intercept=0.5, tKick=1e-3, tTrans=0, noise_drive=True, load=[])