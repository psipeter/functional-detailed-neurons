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

def feedback(x):
    sigma = 10
    beta = 8/3
    rho = 28
    dx = sigma*(x[1] - x[0])
    # dy = x[0] * (rho - x[2]) - x[1]
    dy = -x[1] - x[0]*x[2] + rho*x[0]
    dz = x[0]*x[1] - beta*x[2]
    return [dx, dy, dz]

def fb(x):
    sigma = 10
    beta = 8/3
    rho = 28
    tau = 0.1
    dx0 = -sigma * x[0] + sigma * x[1]
    dx1 = -x[0] * x[2] - x[1]
    dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
    return [
        dx0 * tau + x[0],
        dx1 * tau + x[1],
        dx2 * tau + x[2],
    ]

def makeKick(tKick, seed=seed):
    rng = np.random.RandomState(seed=seed)
    kick = rng.uniform(0,1,size=3)
    stim_func = lambda t: kick if t<tKick else [0,0,0]
    return stim_func

def go(neuron_type, t=10, seed=0, dt=0.001, nBias=100, nPre=1000, nEns=300, r=40, m=Uniform(30,40),
    stim_func=lambda t: 0, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    dB=None, eB=None, wB=None, d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, d2=None, 
    learn0=False, learn1=False, learnB=False, learnDF=False, test=False,
    eRate=1e-6, dRate=3e-7):

    weightsBias = wB if (np.any(wB) and not learnB) else np.zeros((nBias, nEns))
    weightsFF = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weightsFB = w1 if (np.any(w1) and not learn0 and not learn1) else np.zeros((nEns, nEns))
    with nengo.Network(seed=seed) as model:
        const = nengo.Node(1)
        inpt = nengo.Node(stim_func)  # kick
        bias = nengo.Ensemble(nBias, 3, max_rates=m, neuron_type=ReLu(), seed=seed)
        pre = nengo.Ensemble(nPre, 3, max_rates=m, radius=r, neuron_type=ReLu(), seed=seed)
        ens = nengo.Ensemble(nEns, 3, neuron_type=neuron_type, seed=seed)
        tarA = nengo.Ensemble(nEns, 3, max_rates=m, radius=r, neuron_type=ReLu(), seed=seed)
        tarX = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        tarX2 = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())

        nengo.Connection(inpt, tarX, synapse=None)
        nengo.Connection(tarX, tarX, function=feedback, synapse=~s)
        nengo.Connection(tarX, tarX2, synapse=fTarget)
        nengo.Connection(tarX2, tarX3, synapse=fTarget)

        nengo.Connection(bias, tarA, synapse=fTarget, seed=seed)
        connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
        connFF = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weightsFF, weights=True), seed=seed)

        if learnB:  # learn a bias connection that feeds spikes representing zero to all ensembles
            nodeBias = LearningNode(bias, ens, 1, conn=connBias, d=dB, e=eB, w=wB, eRate=eRate, dRate=dRate)
            nengo.Connection(bias.neurons, nodeBias[:nBias], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeBias[nBias: nBias+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeBias[nBias+nEns: nBias+nEns+nEns], synapse=fSmooth)
            nengo.Connection(const, nodeBias[-1:], synapse=fTarget)
            nengo.Connection(nodeBias, ens.neurons, synapse=None)            

        if learn0:  # learn to receive supervised "recurrent" input from ReLU
            nengo.Connection(tarX, pre, synapse=None, seed=seed)
            nengo.Connection(tarX, tarA, synapse=fTarget, seed=seed)
            nodeFF = LearningNode(pre, ens, 3, conn=connFF, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, nodeFF[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeFF[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeFF[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(tarX, nodeFF[-3:], synapse=None)
            nengo.Connection(nodeFF, ens.neurons, synapse=None)

        if learnDF:  # train decoders and filters for ens
            nengo.Connection(tarX, pre, synapse=None, seed=seed)

        if learn1: # learn to receive supervised "recurrent" input from ens
            ens2 = nengo.Ensemble(nEns, 3, neuron_type=neuron_type, seed=seed)
            tarA2 = nengo.Ensemble(nEns, 3, max_rates=m, radius=r, neuron_type=ReLu(), seed=seed)
            nengo.Connection(tarX, pre, synapse=None, seed=seed)
            nengo.Connection(bias, tarA2, synapse=fTarget, seed=seed)
            nengo.Connection(tarX2, tarA2, synapse=fTarget, seed=seed)
            connSupv = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            nodeSupv = LearningNode(ens, ens2, 3, conn=connSupv, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, nodeSupv[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, nodeSupv[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA2.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(nodeSupv, ens2.neurons, synapse=None)

        if test:
            nengo.Connection(inpt, pre, synapse=None, seed=seed)
            connFB = nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(weightsFB, weights=True))  # recurrent

        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None) if learn1 else None
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarA2 = nengo.Probe(tarA2.neurons, synapse=None) if learn1 else None
        pTarX = nengo.Probe(tarX, synapse=None)
        pTarX2 = nengo.Probe(tarX2, synapse=None)
        pTarX3 = nengo.Probe(tarX3, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, NEURON): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, NEURON): nrnReset(sim, model)

    if learnB:
        dB, eB, wB = nodeBias.d, nodeBias.e, nodeBias.w    
    if learn0:
        d0, e0, w0 = nodeFF.d, nodeFF.e, nodeFF.w
    if learn1:
        e1, w1 = nodeSupv.e, nodeSupv.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens=sim.data[pEns],
        ens2=sim.data[pEns2] if learn1 else None,
        tarA=sim.data[pTarA],
        tarA2=sim.data[pTarA2] if learn1 else None,
        tarX=sim.data[pTarX],
        tarX2=sim.data[pTarX2],
        tarX3=sim.data[pTarX3],
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
        f1=f1,
        dB=dB,
        eB=eB,
        wB=wB,
    )

def measure_correlation_integral(X, l, N, seed):
    rng = np.random.RandomState(seed=seed)
    times = rng.choice(range(len(X)), size=(N,2), replace=False)
    Xi = X[times[:,0]]
    Xj = X[times[:,1]]
    delta = np.linalg.norm(Xi-Xj, axis=1)
    n_lesser = len(np.where(delta<l)[0])
    C = 1/np.square(N) * n_lesser
    return l, C

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, tKick=1e0, seed=0, tBias=30,
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), penalty=0, reg=1e-3,
    load=[]):

    print(f'Neuron type: {neuron_type}')

    if 0 in load:
        data = np.load(f"data/lorenz_{neuron_type}.npz")
        dB, eB, wB = data['dB'], data['eB'], data['wB']
    else:
        print('train dB, eB, wB from bias to ens')
        dB, eB, wB = None, None, None
        for n in range(nTrain):
            data = go(neuron_type, learnB=True, eRate=10*eRate,
                nEns=nEns, t=tBias, dt=dt, seed=seed,
                dB=dB, eB=eB, wB=wB,
                fTarget=fTarget, fSmooth=fSmooth)
            dB, eB, wB = data['dB'], data['eB'], data['wB']
            np.savez(f"data/lorenz_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "lorenz", neuron_type, "bias", n, nTrain)

    if 1 in load:
        data = np.load(f"data/lorenz_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func = makeKick(tKick, seed=seed)
            data = go(neuron_type, learn0=True, eRate=eRate, stim_func=stim_func,
                nEns=nEns, t=tTrain, dt=dt, seed=seed,
                wB=wB,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/lorenz_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "lorenz", neuron_type, "ens", n, nTrain)

    if 2 in load:
        data = np.load(f"data/lorenz_{neuron_type}.npz")
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('train d1 and f1 for ens to compute the feedback function')
        targets = np.zeros((nTrain, int(tTrain/dt), 3))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func = makeKick(tKick, seed=seed)
            data = go(neuron_type, learnDF=True, stim_func=stim_func,
                nEns=nEns, t=tTrain, dt=dt, seed=seed,
                wB=wB,
                w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            targets[n] = data['tarX3']
            spikes[n] = data['ens']
        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrain,
            network="lorenz", neuron_type=neuron_type, ens="ens", dt=dt,
            penalty=penalty, seed=seed, reg=reg)
        print(f"taus: {tauRise1:.4f}, {tauFall1:.4f}")
        np.savez(f"data/lorenz_{neuron_type}.npz",
            dB=dB, eB=eB, wB=wB,
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)
        f1 = DoubleExp(tauRise1, tauFall1)
        times = data['times']
        tarX = data['tarX3']
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)

        fig = plt.figure(figsize=((18, 4)))
        ax = fig.add_subplot(111, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax.plot(*tarX.T, linewidth=0.25)
        ax2.plot(*xhat.T, linewidth=0.25)
        ax.set(title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
        ax2.set(title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
        ax.grid(False)
        ax2.grid(False)
        plt.tight_layout()
        plt.savefig(f"plots/lorenz/decode_{neuron_type}.pdf", bbox_inches="tight", pad_inches=0.1)

    if 3 in load:
        data = np.load(f"data/lorenz_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from ens to ens2')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func = makeKick(tKick, seed=seed)
            data = go(neuron_type, learn1=True, eRate=eRate, stim_func=stim_func,
                nEns=nEns, t=tTrain, dt=dt, seed=seed,
                wB=wB,
                w0=w0,
                d1=d1, f1=f1,
                e1=e1, w1=w1,
                fTarget=fTarget, fSmooth=fSmooth)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/lorenz_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
                "lorenz", neuron_type, "ens2", n, nTrain)

        print('check ens to ens2 connection')
        stim_func = makeKick(tKick, seed=seed)
        data = go(neuron_type, learn1=True, eRate=0, stim_func=stim_func,
            nEns=nEns, t=tTrain, dt=dt, seed=seed,
            wB=wB,
            w0=w0,
            d1=d1, f1=f1,
            e1=e1, w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        tarX = fTarget.filt(data['tarX3'], dt=dt)
        aEns2 = f1.filt(data['ens2'], dt=dt)
        xhat = np.dot(aEns2, d1)
        fig = plt.figure(figsize=((18, 4)))
        ax = fig.add_subplot(111, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax.plot(*tarX.T, linewidth=0.25)
        ax2.plot(*xhat.T, linewidth=0.25)
        ax.set(title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
        ax2.set(title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
        ax.grid(False)
        ax2.grid(False)
        plt.tight_layout()
        fig.savefig(f'plots/lorenz/feedforward_{neuron_type}.pdf', bbox_inches="tight", pad_inches=0.1)


    dfs = []
    columns = ('neuron_type', 'n', 'correlation integral slope error')
    print('estimating error')
    for n in range(nTest):
        stim_func = makeKick(tKick, seed=100+seed)
        data = go(neuron_type, test=True, stim_func=stim_func,
            nEns=nEns, t=tTest, dt=dt, seed=seed,
            wB=wB,
            w0=w0,
            f1=f1,
            w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        times = data['times']
        tarX3 = fTarget.filt(data['tarX2'], dt=dt)
        tarX = data['tarX']
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        fig = plt.figure(figsize=((18, 4)))
        ax = fig.add_subplot(111, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax.plot(*tarX.T, linewidth=0.25)
        ax2.plot(*xhat.T, linewidth=0.25)
        ax.set(title='target', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
        ax2.set(title='xhat', xlabel=r"$\mathbf{x}$", ylabel=r"$\mathbf{y}$", zlabel=r"$\mathbf{z}$")
        ax.grid(False)
        ax2.grid(False)
        plt.tight_layout()
        fig.savefig(f'plots/lorenz/test_{neuron_type}_{n}.pdf')

        ls = []
        Cs = []
        for l in [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 10]:
            print(f'l={l}')
            l, C = measure_correlation_integral(xhat, l=l, N=10000, seed=seed)
            if C > 0:
                ls.append(l)
                Cs.append(C)

        slope, intercept, r, p, se = linregress(np.log2(ls), np.log2(Cs))
        error = slope / 2.05
        fig, ax = plt.subplots()
        ax.scatter(np.log2(ls), np.log2(Cs))
        ax.plot(np.log2(ls), intercept+slope*np.log2(ls), label=f'slope={slope:.3}, r={r:.3}, p={p:.3f}')
        ax.set(xlabel='log2(l)', ylabel='log2(C)')
        ax.legend()
        fig.savefig(f"plots/lorenz/correlation_integrals_{neuron_type}_{n}.pdf")
        plt.close('all')

        # dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, error]], columns=columns))

    # return times, tarX, xhat, dfs

def compare(neuron_types, nTrain=3, tTrain=100, nTest=3, tTest=100, tKick=1e-2, load=[],
    eRates=[1e-7, 1e-5, 1e-6, 1e-7]):

    dfsAll = []
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=((7, 4)))
    for i, neuron_type in enumerate(neuron_types):
        run(neuron_type, nTrain, nTest, tTrain, tTest, tKick=tKick, eRate=eRates[i], load=load)

compare([LIF()], load=[0,1])