import numpy as np
import nengo

class LearningNode(nengo.Node):
    def __init__(self, pre, post, tar, dim, conn, w=None, e=None, d=None, e_rate=1e-6, d_rate=1e-6, exc=False, inh=False):
        self.pre = pre
        self.post = post
        self.tar = tar
        self.conn = conn  # facilitates NEURON connections
        self.size_in = pre.n_neurons + post.n_neurons + tar.n_neurons + dim
        self.size_out = post.n_neurons
        # 'encoders' is a connection-specific tensor used to compute weights; shape (Npre, Npost, d)
        self.d = d if np.any(d) else np.zeros((pre.n_neurons, dim))
        self.e = e if np.any(e) else np.zeros((pre.n_neurons, post.n_neurons, dim))
        self.w = w if np.any(w) else np.zeros((pre.n_neurons, post.n_neurons))
        self.e_rate = e_rate
        self.d_rate = d_rate
        self.exc = exc
        self.inh = inh
        assert self.exc==False or self.inh==False, "Can't force excitatory and inhibitory weights"
        super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)

    def step(self, t, x):
        aPre = x[:self.pre.n_neurons]  # presynaptic activities
        aPost = x[self.pre.n_neurons: self.pre.n_neurons+self.post.n_neurons]  # postsynaptic activities
        aTar = x[self.pre.n_neurons+self.post.n_neurons: self.pre.n_neurons+self.post.n_neurons+self.tar.n_neurons]  # target activities
        xTar = x[self.pre.n_neurons+self.post.n_neurons+self.tar.n_neurons:]  # state-space target for decoder update
        '''decoder update'''
        xPre = np.dot(aPre, self.d)
        aPre2 = aPre.reshape((-1, 1))
        xError = (xTar - xPre).reshape((-1, 1))
        self.d += self.d_rate / self.pre.n_neurons * np.dot(aPre2, xError.T)
        '''encoder update, thanks to Andreas'''
        self.e += self.e_rate * np.einsum("n,m,nd->nmd", aPre, aTar-aPost, np.sign(self.d))
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
#                         dE = sign * self.e_rate * aPre[pre] # delta for that entry in the encoder matrix
#                         self.e[pre, post, dim] += dA * dE
#                     w = np.dot(self.d[pre], self.e[pre, post]) # update corresponding entry in weight matrix
#                     if self.exc and w < 0: w = 0 # enforce excitatory weights
#                     if self.inh and w > 0: w = 0 # enforce inhibitory weights
#                     self.w[pre, post] = w # update weight matrix
        return np.dot(self.w.T, aPre)  # transmit weighted activity from pre to post



def trainDF(spikes, targets, nTrain,
    dt=0.001, dtSample=0.001, reg=1e-3, penalty=0, evals=100, name="default",
    tauRiseMin=1e-3, tauRiseMax=3e-2, tauFallMin=1e-2, tauFallMax=3e-1):

    np.savez_compressed('data/%s_spikes.npz'%name, spikes=spikes)
    np.savez_compressed('data/%s_target.npz'%name, target=target)
    hyperparams = {}
    hyperparams['nTrain'] = nTrain
    hyperparams['name'] = name
    hyperparams['reg'] = reg
    hyperparams['dt'] = dt
    hyperparams['dtSample'] = dtSample
    hyperparams['tauRise'] = hp.uniform('tauRise', tauRiseMin, tauRiseMax)
    hyperparams['tauFall'] = hp.uniform('tauFall', tauFallMin, tauFallMax)

    def objective(hyperparams):
        tauRise = hyperparams['tauRise']
        tauFall = hyperparams['tauFall']
        dt = hyperparams['dt']
        dtSample = hyperparams['dtSample']
        f = DoubleExp(tauRise, tauFall)
        spikes = np.load('data/%s_spikes.npz'%hyperparams['name'])['spikes']
        targets = np.load('data/%s_target.npz'%hyperparams['name'])['target']
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