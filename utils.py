import numpy as np
import nengo
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, base
from nengolib import Lowpass, DoubleExp
from nengo.solvers import LstsqL2
from nengo.utils.numpy import rmse

class LearningNode(nengo.Node):
    def __init__(self, pre, post, dim, conn, w=None, e=None, d=None, eRate=1e-6, dRate=3e-6, exc=False, inh=False):
        self.pre = pre
        self.post = post
        self.conn = conn  # facilitates NEURON connections
        self.size_in = pre.n_neurons + post.n_neurons + post.n_neurons + dim
        self.size_out = post.n_neurons
        # 'encoders' is a connection-specific tensor used to compute weights; shape (Npre, Npost, d)
        self.d = d if np.any(d) else np.zeros((pre.n_neurons, dim))
        self.e = e if np.any(e) else np.zeros((pre.n_neurons, post.n_neurons, dim))
        self.w = w if np.any(w) else np.zeros((pre.n_neurons, post.n_neurons))
        self.eRate = eRate
        self.dRate = dRate
        self.exc = exc
        self.inh = inh
        assert self.exc==False or self.inh==False, "Can't force excitatory and inhibitory weights"
        super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)

    def step(self, t, x):
        aPre = x[:self.pre.n_neurons]  # presynaptic activities
        aPost = x[self.pre.n_neurons: self.pre.n_neurons+self.post.n_neurons]  # postsynaptic activities
        aTar = x[self.pre.n_neurons+self.post.n_neurons: self.pre.n_neurons+self.post.n_neurons+self.post.n_neurons]  # target activities
        xTar = x[self.pre.n_neurons+self.post.n_neurons+self.post.n_neurons:]  # state-space target for decoder update
        '''decoder update'''
        xPre = np.dot(aPre, self.d)
        aPre2 = aPre.reshape((-1, 1))
        xError = (xTar - xPre).reshape((-1, 1))
        self.d += self.dRate / self.pre.n_neurons * np.dot(aPre2, xError.T)
        '''encoder update, thanks to Andreas'''
        self.e += self.eRate * np.einsum("n,m,nd->nmd", aPre, aTar-aPost, np.sign(self.d))
        self.w = np.einsum("nd,nmd->nm", self.d, self.e)
        if self.exc: self.w = np.clip(self.w, 0, np.inf) # enforce excitatory weights
        if self.inh: self.w = np.clip(self.w, -np.inf, 0) # enforce inhibitory weights
        if hasattr(self.conn, 'w'): self.conn.w = self.w  # update NEURON objects
        '''Equivalent implementation for encoder update using loops, for reference/readability'''
#             for post in range(self.post.n_neurons):
#                 dA = aTar[post] - aPost[post]  # delta used for update
#                 if np.abs(dA)<dAmin: continue  # skip update if post activity is close to target activity
#                 for pre in range(self.pre.n_neurons):
#                     for dim in range(self.d.shape[1]): # each dimension of the 'encoder' is updated separately
#                         sign = 1.0 if self.d[pre, dim] >= 0 else -1.0 # sign ensures weight change is in the correct direction
#                         dE = sign * self.eRate * aPre[pre] # delta for that entry in the encoder matrix
#                         self.e[pre, post, dim] += dA * dE
#                     w = np.dot(self.d[pre], self.e[pre, post]) # update corresponding entry in weight matrix
#                     if self.exc and w < 0: w = 0 # enforce excitatory weights
#                     if self.inh and w > 0: w = 0 # enforce inhibitory weights
#                     self.w[pre, post] = w # update weight matrix
        return np.dot(self.w.T, aPre)  # transmit weighted activity from pre to post



def trainDF(spikes, targets, nTrain, network, neuron_type, ens,
    dt=0.001, dtSample=0.001, reg=1e-3, penalty=0, evals=100, seed=0,
    tauRiseMin=1e-3, tauRiseMax=3e-2, tauFallMin=1e-2, tauFallMax=3e-1):

    np.savez_compressed(f'data/{network}_{neuron_type}_{ens}_spikes.npz', spikes=spikes)
    np.savez_compressed(f'data/{network}_{neuron_type}_{ens}_target.npz', targets=targets)
    hyperparams = {}
    hyperparams['nTrain'] = nTrain
    hyperparams['network'] = network
    hyperparams['neuron_type'] = neuron_type
    hyperparams['ens'] = ens
    hyperparams['reg'] = reg
    hyperparams['dt'] = dt
    hyperparams['dtSample'] = dtSample
    hyperparams['tauRise'] = hp.uniform('tauRise', tauRiseMin, tauRiseMax)
    hyperparams['tauFall'] = hp.uniform('tauFall', tauFallMin, tauFallMax)

    def objective(hyperparams):
        network = hyperparams['network']
        neuron_type = hyperparams['neuron_type']
        ens = hyperparams['ens']
        tauRise = hyperparams['tauRise']
        tauFall = hyperparams['tauFall']
        dt = hyperparams['dt']
        dtSample = hyperparams['dtSample']
        f = DoubleExp(tauRise, tauFall)
        spikes = np.load(f'data/{network}_{neuron_type}_{ens}_spikes.npz')['spikes']
        targets = np.load(f'data/{network}_{neuron_type}_{ens}_target.npz')['targets']
        A = np.zeros((0, spikes.shape[2]))
        Y = np.zeros((0, targets.shape[2]))
        for n in range(hyperparams['nTrain']):
            A = np.append(A, f.filt(spikes[n], dt=dt), axis=0)
            Y = np.append(Y, targets[n], axis=0)
        if dt != dtSample:
            A = A[::int(dtSample/dt)]
            Y = Y[::int(dtSample/dt)]
        d, _ = LstsqL2(reg=hyperparams['reg'])(A, Y)
        X = np.dot(A, d)
        loss = rmse(X, Y)
        loss += penalty * (10*tauRise + tauFall)
        return {'loss': loss, 'd': d, 'tauRise': tauRise, 'tauFall': tauFall, 'status': STATUS_OK}
    
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials)
    idx = np.argmin(trials.losses())
    best = trials.trials[idx]
    d = best['result']['d']
    tauRise = best['result']['tauRise']
    tauFall = best['result']['tauFall']

    return d, tauRise, tauFall

def trainD(spikes, targets, nTrain, f, dt=0.001, reg=1e-3):
    A = np.zeros((0, spikes.shape[2]))
    Y = np.zeros((0, targets.shape[2]))
    for n in range(nTrain):
        A = np.append(A, f.filt(spikes[n], dt=dt), axis=0)
        Y = np.append(Y, targets[n], axis=0)
    d, _ = LstsqL2(reg=reg)(A, Y)
    return d

def fitSinusoid(times, xhat, tTrans, muFreq=1, sigmaFreq=0.1, base=False, name="sinusoid", seed=0, evals=100):
    np.savez_compressed('data/%s_times.npz'%name, times=times)
    np.savez_compressed('data/%s_vals.npz'%name, xhat=xhat)
    hyperparams = {}
    hyperparams['name'] = name
    hyperparams['tTrans'] = tTrans
    hyperparams['freq'] = hp.normal('freq', muFreq, sigmaFreq)
    hyperparams['phase'] = hp.uniform('phase', 0, 2*np.pi)
#     hyperparams['mag'] = 0.75
#     hyperparams['base'] = 0
#     hyperparams['phase'] = hp.choice('phase', [0, 0.2, 0.4, 0.6, 0.8])
#     hyperparams['freq'] = hp.uniform('freq', fMin*freq, fMax*freq)
    hyperparams['mag'] = hp.choice('mag', [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    hyperparams['base'] = hp.choice('base', [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]) if base else 0
#     hyperparams['mag'] = hp.uniform('mag', 0.7, 1.0)
#     hyperparams['base'] = hp.uniform('base', -0.2, 0.2)
#     hyperparams['mag'] = 0.8

    def objective(hyperparams):
        freq = hyperparams['freq']
        tTrans = hyperparams['tTrans']
        phase = hyperparams['phase']
        mag = hyperparams['mag']
        base = hyperparams['base']
        times = np.load('data/%s_times.npz'%hyperparams['name'])['times']
        sin = base + mag*np.sin(times*2*np.pi*freq+phase)
        xhat = np.load('data/%s_vals.npz'%hyperparams['name'])['xhat']
        loss = rmse(sin[tTrans:], xhat[tTrans:])
        return {'loss': loss, 'freq': freq, 'phase': phase, 'mag': mag, 'base': base, 'status': STATUS_OK}
    
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=tpe.suggest,
#         algo=hyperopt.rand.suggest,
        max_evals=evals,
        trials=trials)
    idx = np.argmin(trials.losses())
    best = trials.trials[idx]
    freq = best['result']['freq']
    phase = best['result']['phase']
    mag = best['result']['mag']
    base = best['result']['base']
        
    return freq, phase, mag, base
