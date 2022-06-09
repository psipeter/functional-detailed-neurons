import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from nengolib.signal import s
from utils import trainDF
from nengolib import Lowpass, DoubleExp
from nengo.solvers import NoSolver, LstsqL2

def measure_correlation_integral(X, l, N, rng):
	n_lesser = 0
	times = rng.choice(range(len(X)), size=(N,2), replace=False)
	Xi = X[times[:,0]]
	Xj = X[times[:,1]]
	delta = np.linalg.norm(Xi-Xj, axis=1)
	n_lesser = len(np.where(delta<l)[0])
	C = 1/np.square(N) * n_lesser
	return l, C

sigma = 10
beta = 8.0 / 3
rho = 28
tau = 0.1

def feedback(x):
	dx0 = -sigma * x[0] + sigma * x[1]
	dx1 = -x[0] * x[2] - x[1]
	dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
	return [dx0 * tau + x[0], dx1 * tau + x[1], dx2 * tau + x[2]]

def feedback_tar(x):
	dx = sigma*(x[1] - x[0])
	# dy = x[0] * (rho - x[2]) - x[1]
	dy = -x[1] - x[0]*x[2] + rho*x[0]
	dz = x[0]*x[1] - beta*x[2]
	return [dx, dy, dz]

def run(
		nEns = 1000,
		r = 50,
		neuron_type = nengo.LIF(),
		seed = 0,
		t = 100,
		nTrain = 1,
		tTrans = 0,
		evals = 10,
		dt = 1e-3,
		reg = 1e-2,
		penalty = 0,
		fTarget = Lowpass(0.1),
		tKick = 1e0,
		optimize = 'df',
		function = 'tar',
		filter_target = False,
		max_rates = nengo.dists.Uniform(30, 40)
	):

	rng = np.random.RandomState(seed=seed)

	def go(phase, d, f):
		with nengo.Network(seed=seed) as model:
			inpt = nengo.Node(lambda t: rng.uniform(0,1,size=3) if t<tKick else [0,0,0])  # kick
			tar = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
			tar2 = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
			ens = nengo.Ensemble(nEns, 3, max_rates=max_rates, radius=r, neuron_type=neuron_type, seed=seed)
			nengo.Connection(inpt, ens, synapse=None)
			nengo.Connection(inpt, tar, synapse=None)
			nengo.Connection(tar, tar, function=feedback_tar, synapse=~s)
			nengo.Connection(tar, tar2, function=feedback, synapse=None)
			if phase=="train":
				nengo.Connection(tar, ens, synapse=None)
			elif phase=="test":
				nengo.Connection(ens.neurons, ens, transform=d.T, synapse=f)        
			spk = nengo.Probe(ens.neurons, synapse=None)
			ptar = nengo.Probe(tar, synapse=None)
			ptar2 = nengo.Probe(tar2, synapse=None)
		with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
			sim.run(t, progress_bar=True)
		return sim.data[spk], sim.data[ptar], sim.data[ptar2]

	if optimize=='df':
		targets = np.zeros((nTrain, int((t-tTrans)/dt), 3))
		spikes = np.zeros((nTrain, int((t-tTrans)/dt), nEns))
		for n in range(nTrain):
			spk, tar, tar2 = go("train", None, None)
			if function=='tar':
				if filter_target:
					targets[n] = fTarget.filt(tar, dt=dt)[int(tTrans/dt):]
				else:
					targets[n] = tar[int(tTrans/dt):]
			if function=='tar2':
				if filter_target:
					targets[n] = fTarget.filt(tar2, dt=dt)[int(tTrans/dt):]
				else:
					targets[n] = tar2[int(tTrans/dt):]
			spikes[n] = spk[int(tTrans/dt):]

		d, rise, fall = trainDF(spikes, targets, 1, network="lorenz", neuron_type="ReLU()", ens="nef", dt=dt,
			# tauRiseMin=1e-3, tauRiseMax=3e-3, tauFallMin=1e-3, tauFallMax=2e-2,
			penalty=penalty, seed=seed, reg=reg, evals=evals)
		f = DoubleExp(rise, fall)
		print('synapse', rise, fall)            
	elif optimize=='d':
		f = fTarget
		spk, tar, tar2 = go("train", None, None)
		act = f.filt(spk, dt=dt)
		if function=='tar':
			targets = tar
		elif function=='tar2':
			targets = tar2
		d = LstsqL2(reg=reg)(act, targets)[0]
	elif optimize=='nef':
		f = fTarget
		with nengo.Network(seed=seed) as model:
			ens = nengo.Ensemble(nEns, 3, max_rates=max_rates, radius=r, neuron_type=neuron_type, seed=seed)
			c = nengo.Connection(ens, ens, function=feedback, synapse=f)
		with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
			d = sim.data[c].weights.T
			
	spk, tar, tar2 = go("test", d, f)
	act = f.filt(spk, dt=dt)
	xhat = np.dot(act, d)
	target = fTarget.filt(tar, dt=dt)
		
	fig = plt.figure(figsize=((12, 6)))
	ax = fig.add_subplot(121, projection='3d')
	ax2 = fig.add_subplot(122, projection='3d')
	ax.plot(*xhat.T, linewidth=0.25)
	ax2.plot(*target.T, linewidth=0.25)
	ax.set(title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
	ax2.set(title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
	ax.grid(False)
	ax2.grid(False)
	plt.tight_layout()
	fig.savefig(f"plots/lorenz/fiddle_state.pdf")

	rng = np.random.RandomState(seed=seed)
	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((12, 6)))
	ls = []
	Cs = []
	for l in [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10]:
		l, C = measure_correlation_integral(xhat, l=l, N=10000, rng=rng)
		if C > 0:
			ls.append(l)
			Cs.append(C)
	slope, intercept, r, p, se = linregress(np.log2(ls), np.log2(Cs))
	error = slope / 2.05
	ax.scatter(np.log2(ls), np.log2(Cs))
	ax.plot(np.log2(ls), intercept+slope*np.log2(ls), label=f'slope={slope:.3}, r={r:.3}, p={p:.3f}')
	ax.set(xlabel='log2(l)', ylabel='log2(C)', title='xhat')
	ax.legend()
	ls = []
	Cs = []
	for l in [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10]:
		l, C = measure_correlation_integral(target, l=l, N=10000, rng=rng)
		if C > 0:
			ls.append(l)
			Cs.append(C)
	slope, intercept, r, p, se = linregress(np.log2(ls), np.log2(Cs))
	error = slope / 2.05
	ax2.scatter(np.log2(ls), np.log2(Cs))
	ax2.plot(np.log2(ls), intercept+slope*np.log2(ls), label=f'slope={slope:.3}, r={r:.3}, p={p:.3f}')
	ax2.set(xlabel='log2(l)', ylabel='log2(C)', title='target')
	ax2.legend()
	fig.savefig(f"plots/lorenz/fiddle_correlation.pdf")

	max_rates = np.max(act, axis=0)
	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((12, 6)))
	sns.histplot(max_rates, stat='percent', ax=ax)
	sns.histplot(d.ravel(), stat='percent', ax=ax2)
	ax.set(title='firing rates')
	ax2.set(title='decoder')
	fig.savefig(f"plots/lorenz/fiddle_histograms.pdf")

run(nEns=300, t=200, r=40, nTrain=10, evals=20, max_rates=nengo.dists.Uniform(30, 40),
	fTarget=Lowpass(1e-2), filter_target=True, optimize="df", function='tar')