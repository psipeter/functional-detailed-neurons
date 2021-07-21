import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp

from neuron_types import LIF, Izhikevich, Wilson, Pyramidal, nrnReset
from utils import LearningNode, trainDF
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

def makeSignal(t, dt=0.001, value=1, seed=0):
    stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        inpt = nengo.Node(stim)
        probe = nengo.Probe(inpt, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = sim.data[probe]
    if np.abs(np.max(u)) > np.abs(np.min(u)):
        stim = u * value / np.max(u)
        if seed%2==0: stim*=-1
    else:
        stim = u / np.min(u)
        if seed%2==0: stim*=-1
    stim_func = lambda t: stim[int(t/dt)]
    return stim_func

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=10,
    m=Uniform(20, 40), stim_func=lambda t: 0,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, learn0=False, learn1=False,
    eRate=1e-6, dRate=3e-6):

    weights0 = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weights1 = w1 if (np.any(w1) and not learn1) else np.zeros((nEns, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        tarX1 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarX2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarA1 = nengo.Ensemble(nEns, 1, max_rates=m, neuron_type=ReLu(), seed=seed)
        tarA2 = nengo.Ensemble(nEns, 1, max_rates=m, neuron_type=ReLu(), seed=seed+1)
        pre = nengo.Ensemble(nPre, 1, max_rates=m, seed=seed)
        ens1 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed+1)

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarX1, synapse=fTarget)
        nengo.Connection(tarX1, tarX2, synapse=fTarget)
        nengo.Connection(inpt, tarA1, synapse=fTarget)
        nengo.Connection(tarX1, tarA2, synapse=fTarget)
        # nengo.Connection(ens1, tarA2, synapse=f1, solver=NoSolver(d1, weights=False))
        conn0 = nengo.Connection(pre, ens1, synapse=fTarget, solver=NoSolver(weights0, weights=True))
        conn1 = nengo.Connection(ens1, ens2, synapse=f1, solver=NoSolver(weights1, weights=True))

        if learn0:
            node0 = LearningNode(pre, ens1, 1, conn=conn0, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, node0[:nPre], synapse=fTarget)
            nengo.Connection(ens1.neurons, node0[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA1.neurons, node0[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, node0[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(node0, ens1.neurons, synapse=None)
        if learn1:
            node1 = LearningNode(ens1, ens2, 1, conn=conn1, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            # node1 = LearningNode(ens1, ens2, 1, conn=conn1, d=d1, e=e1, w=w1, eRate=eRate, dRate=dRate)
            nengo.Connection(ens1.neurons, node1[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, node1[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA2.neurons, node1[2*nEns: 3*nEns], synapse=fSmooth)
            # nengo.Connection(tarX1, node1[3*nEns:], synapse=None)
            nengo.Connection(node1, ens2.neurons, synapse=None)

        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns1 = nengo.Probe(ens1.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTarA1 = nengo.Probe(tarA1.neurons, synapse=None)
        pTarA2 = nengo.Probe(tarA2.neurons, synapse=None)
        pTarX1 = nengo.Probe(tarX1, synapse=None)
        pTarX2 = nengo.Probe(tarX2, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Pyramidal): neuron.h.init()  # broken??
        # if isinstance(neuron_type, Pyramidal): neuron.h.__init__()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)
    
    if learn0:
        d0, e0, w0 = node0.d, node0.e, node0.w
    if learn1:
        e1, w1 = node1.e, node1.w
        # d1, e1, w1 = node1.d, node1.e, node1.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens1=sim.data[pEns1],
        ens2=sim.data[pEns2],
        tarA1=sim.data[pTarA1],
        tarA2=sim.data[pTarA2],
        tarX1=sim.data[pTarX1],
        tarX2=sim.data[pTarX2],
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
    )

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate,
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), load=[]):

    print(f'Neuron type: {neuron_type}')
    if 0 in load:
        data = np.load(f"data/identity_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens1')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.4, dt=dt, seed=n)
            data = go(neuron_type, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/identity_{neuron_type}.npz", d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens1'], dt=dt), fSmooth.filt(data['tarA1'], dt=dt),
                "identity", neuron_type, "ens1", n, nTrain)

    if 1 in load:
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('train d1 and f1 for ens1 to compute identity')
        targets = np.zeros((nTrain, int(tTrain/dt), 1))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.4, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            targets[n] = fTarget.filt(data['tarX1'], dt=dt)
            spikes[n] = data['ens1']

        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrain, dt=dt, network="identity", neuron_type=neuron_type, ens="ens1")
        f1 = DoubleExp(tauRise1, tauFall1)
        np.savez(f"data/identity_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)

    if 2 in load:
        data = np.load(f"data/identity_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
        # d1, e1, w1 = data['d1'], data['e1'], data['w1']
    else:
        print('train e1, w1 from ens1 to ens2')
        e1, w1 = None, None
        # d1, e1, w1 = None, None, None
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.4, dt=dt, seed=n)
            data = go(neuron_type, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            e1, w1 = data['e1'], data['w1']
            # d1, e1, w1 = data['d1'], data['e1'], data['w1']
            np.savez(f"data/identity_{neuron_type}.npz",
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
                "identity", neuron_type, "ens2", n, nTrain)

    if 3 in load:
        d2, tauRise2, tauFall2 = data['d2'], data['tauRise2'], data['tauFall2']
        f2 = DoubleExp(tauRise2, tauFall2)
    else:
        print('train d2 and f2 for ens2 for readout')
        targets = np.zeros((nTrain, int(tTrain/dt), 1))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.4, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            targets[n] = fTarget.filt(data['tarX2'], dt=dt)
            spikes[n] = data['ens2']

        d2, tauRise2, tauFall2 = trainDF(spikes, targets, nTrain, dt=dt, network="identity", neuron_type=neuron_type, ens="ens2")
        f2 = DoubleExp(tauRise2, tauFall2)
        np.savez(f"data/identity_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
            e1=e1, w1=w1,
            d2=d2, tauRise2=tauRise2, tauFall2=tauFall2)

    dfs = []
    columns = ('neuron_type', 'n', 'error1', 'error2')
    print('estimating error')
    for n in range(nTest):
        stim_func = makeSignal(tTest, dt=dt, seed=100+n)
        data = go(neuron_type,
            nEns=nEns, t=tTest, dt=dt,
            d0=d0, e0=e0, w0=w0,
            d1=d1, e1=e1, w1=w1, f1=f1,
            fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
        times = data['times']
        tarX1 = fTarget.filt(data['tarX1'], dt=dt)
        tarX2 = fTarget.filt(data['tarX2'], dt=dt)
        aEns1 = f1.filt(data['ens1'], dt=dt)
        aEns2 = f2.filt(data['ens2'], dt=dt)
        xhat1 = np.dot(aEns1, d1)
        xhat2 = np.dot(aEns2, d2)
        error1 = rmse(xhat1, tarX1)
        error2 = rmse(xhat2, tarX2)
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, error1, error2]], columns=columns))

    return times, tarX1, tarX2, xhat1, xhat2, dfs

def compare(neuron_types, eRates=[3e-7, 3e-6, 3e-7, 1e-7], nTrain=10, tTrain=10, nTest=10, tTest=10, load=[]):

    dfsAll = []
    columns = ('neuron_type', 'n', 'error1', 'error2')
    fig, ax = plt.subplots(figsize=((5.25, 1.5)))
    fig2, ax2 = plt.subplots(figsize=((5.25, 1.5)))
    for i, neuron_type in enumerate(neuron_types):
        times, tarX1, tarX2, xhat1, xhat2, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest, eRate=eRates[i], load=load)
        dfsAll.extend(dfs)
        ax.plot(times, xhat1, label=f"{str(neuron_type)[:-2]}", linewidth=0.5)
        ax2.plot(times, xhat2, label=f"{str(neuron_type)[:-2]}", linewidth=0.5)
    df = pd.concat([df for df in dfsAll], ignore_index=True)

    ax.plot(times, tarX1, label='target', color='k', linewidth=0.5)
    ax.set(xlim=((0, tTest)), xticks=(()), ylim=((-1, 1)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig.savefig('plots/figures/identity_ens1.pdf')
    fig.savefig('plots/figures/identity_ens1.svg')

    ax2.plot(times, tarX2, label='target', color='k', linewidth=0.5)
    ax2.set(xlim=((0, tTest)), xticks=(()), ylim=((-1, 1)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax2.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig2.savefig('plots/figures/identity_ens2.pdf')
    fig2.savefig('plots/figures/identity_ens2.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='neuron_type', y='error2', ax=ax)
    ax.set(xlabel='', ylim=((0, 0.04)), yticks=((0, 0.04)), ylabel='Error')
    plt.tight_layout()
    fig.savefig('plots/figures/identity_barplot.pdf')
    fig.savefig('plots/figures/identity_barplot.svg')

compare([LIF(), Izhikevich(), Wilson(), Pyramidal()], load=[0,1,2,3])