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
sns.set(context='paper', style='white', font='CMU Serif')


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
    else:
        stim = u / np.min(u)
    u1 = np.sqrt(np.square(stim))
    u2 = np.sqrt(np.square(stim))
    # the multiplied signal should alternate (across seeds)
    # between having an extrema at -1 and 1 (to cover all phase space),
    # so multiply one or both of the inputs by -1 to achieve this
    if seed%4==0:
        pass
    if seed%4==1:
        u1*=-1
    if seed%4==2:
        u2*=-1
    if seed%4==3:
        u1*=-1
        u2*=-1
    stim_func1 = lambda t: u1[int(t/dt)]
    stim_func2 = lambda t: u2[int(t/dt)]
    return stim_func1, stim_func2

def multiply(x):
    return x[0]*x[1]

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=10,
    m=Uniform(20, 40), stim_func1=lambda t: 0, stim_func2=lambda t: 0,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, learn0=False, learn1=False,
    eRate=1e-6, dRate=3e-6):

    weights0 = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weights1 = w1 if (np.any(w1) and not learn1) else np.zeros((nEns, nEns))
    with nengo.Network() as model:
        inpt1 = nengo.Node(stim_func1)
        inpt2 = nengo.Node(stim_func2)
        tarX1 = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarX2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarA1 = nengo.Ensemble(nEns, 2, max_rates=m, neuron_type=ReLu(), seed=seed)
        tarA2 = nengo.Ensemble(nEns, 1, max_rates=m, neuron_type=ReLu(), seed=seed+1)
        pre = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        ens1 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed+1)

        nengo.Connection(inpt1, pre[0], synapse=None)
        nengo.Connection(inpt2, pre[1], synapse=None)
        nengo.Connection(inpt1, tarX1[0], synapse=fTarget)
        nengo.Connection(inpt2, tarX1[1], synapse=fTarget)
        nengo.Connection(tarX1, tarX2, synapse=fTarget, function=multiply)
        nengo.Connection(inpt1, tarA1[0], synapse=fTarget)
        nengo.Connection(inpt2, tarA1[1], synapse=fTarget)
        nengo.Connection(tarX1, tarA2, synapse=fTarget, function=multiply)
        # nengo.Connection(ens1, tarA2, synapse=f1, function=multiply, solver=NoSolver(d1, weights=False))
        conn0 = nengo.Connection(pre, ens1, synapse=fTarget, solver=NoSolver(weights0, weights=True))
        conn1 = nengo.Connection(ens1, ens2, synapse=f1, function=multiply, solver=NoSolver(weights1, weights=True))

        if learn0:
            node0 = LearningNode(pre, ens1, 2, conn=conn0, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, node0[:nPre], synapse=fTarget)
            nengo.Connection(ens1.neurons, node0[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA1.neurons, node0[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt1, node0[-2], synapse=None)
            nengo.Connection(inpt2, node0[-1], synapse=None)
            nengo.Connection(node0, ens1.neurons, synapse=None)
        if learn1:
            node1 = LearningNode(ens1, ens2, 1, conn=conn1, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            # node1 = LearningNode(ens1, ens2, 1, conn=conn1, d=d1, e=e1, w=w1, eRate=eRate, dRate=dRate)
            nengo.Connection(ens1.neurons, node1[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, node1[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA2.neurons, node1[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(node1, ens2.neurons, synapse=None)

        pInpt1 = nengo.Probe(inpt1, synapse=None)
        pInpt2 = nengo.Probe(inpt2, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns1 = nengo.Probe(ens1.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTarA1 = nengo.Probe(tarA1.neurons, synapse=None)
        pTarA2 = nengo.Probe(tarA2.neurons, synapse=None)
        pTarX1 = nengo.Probe(tarX1, synapse=None)
        pTarX2 = nengo.Probe(tarX2, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Pyramidal): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)
    
    if learn0:
        d0, e0, w0 = node0.d, node0.e, node0.w
    if learn1:
        e1, w1 = node1.e, node1.w
        # d1, e1, w1 = node1.d, node1.e, node1.w

    return dict(
        times=sim.trange(),
        inpt1=sim.data[pInpt1],
        inpt2=sim.data[pInpt2],
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
        data = np.load(f"data/multiply_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens1')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, value=1.2, dt=dt, seed=n)
            data = go(neuron_type, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth,
                stim_func1=stim_func1, stim_func2=stim_func2)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/multiply_{neuron_type}.npz", d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens1'], dt=dt), fSmooth.filt(data['tarA1'], dt=dt),
                "multiply", neuron_type, "ens1", n, nTrain)

    if 1 in load:
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('train d1 and f1 for ens1')
        targets = np.zeros((nTrain, int(tTrain/dt), 1))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, value=1.2, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth,
                stim_func1=stim_func1, stim_func2=stim_func2)
            # targets[n] = fTarget.filt(data['tarX1'], dt=dt)
            targets[n] = data['tarX2']  # tarX1[0] * tarX1[1], filtered by fTarget
            spikes[n] = data['ens1']

        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrain, dt=dt, network="multiply", neuron_type=neuron_type, ens="ens1")
        f1 = DoubleExp(tauRise1, tauFall1)
        np.savez(f"data/multiply_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)

    if 2 in load:
        data = np.load(f"data/multiply_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from ens1 to ens2')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, value=1.2, dt=dt, seed=n)
            data = go(neuron_type, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                fTarget=fTarget, fSmooth=fSmooth,
                stim_func1=stim_func1, stim_func2=stim_func2)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/multiply_{neuron_type}.npz",
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
                "multiply", neuron_type, "ens2", n, nTrain)

    if 3 in load:
        d2, tauRise2, tauFall2 = data['d2'], data['tauRise2'], data['tauFall2']
        f2 = DoubleExp(tauRise2, tauFall2)
    else:
        print('train d2 and f2 for ens2')
        targets = np.zeros((nTrain, int(tTrain/dt), 1))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, value=1.2, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                fTarget=fTarget, fSmooth=fSmooth,
                stim_func1=stim_func1, stim_func2=stim_func2)

            targets[n] = fTarget.filt(data['tarX2'], dt=dt)
            spikes[n] = data['ens2']

        d2, tauRise2, tauFall2 = trainDF(spikes, targets, nTrain, dt=dt, network="multiply", neuron_type=neuron_type, ens="ens2")
        f2 = DoubleExp(tauRise2, tauFall2)
        np.savez(f"data/multiply_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
            e1=e1, w1=w1,
            d2=d2, tauRise2=tauRise2, tauFall2=tauFall2)

    dfs = []
    columns = ('neuron_type', 'n', 'error1', 'error2')
    print('estimating error')
    for n in range(nTest):
        stim_func1, stim_func2 = makeSignal(tTest, value=1, dt=dt, seed=100+n)
        data = go(neuron_type,
            nEns=nEns, t=tTest, dt=dt,
            d0=d0, e0=e0, w0=w0,
            d1=d1, e1=e1, w1=w1, f1=f1,
            fTarget=fTarget, fSmooth=fSmooth,
            stim_func1=stim_func1, stim_func2=stim_func2)

        times = data['times']
        tarX1 = data['tarX2']
        tarX2 = fTarget.filt(data['tarX2'], dt=dt)
        aEns1 = f1.filt(data['ens1'], dt=dt)
        aEns2 = f2.filt(data['ens2'], dt=dt)
        xhat1 = np.dot(aEns1, d1)
        xhat2 = np.dot(aEns2, d2)
        error1 = rmse(xhat1, tarX1)
        error2 = rmse(xhat2, tarX2)
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, error1, error2]], columns=columns))

    return times, tarX1, tarX2, xhat1, xhat2, dfs

def compare(neuron_types, eRates=[3e-7, 3e-6, 3e-7, 1e-7], nTrain=30, tTrain=5, nTest=10, tTest=5, load=[]):

    dfsAll = []
    columns = ('neuron_type', 'n', 'error1', 'error2')
    fig1, axes1 = plt.subplots(nrows=len(neuron_types)+1, ncols=1, figsize=((6, len(neuron_types))), sharex=True)
    fig2, axes2 = plt.subplots(nrows=len(neuron_types)+1, ncols=1, figsize=((6, len(neuron_types))), sharex=True)
    for i, neuron_type in enumerate(neuron_types):
        times, tarX1, tarX2, xhat1, xhat2, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest, eRate=eRates[i], load=load)
        dfsAll.extend(dfs)
        axes1[i+1].plot(times, xhat1)
        axes2[i+1].plot(times, xhat2)
        axes1[i+1].set(ylabel=f"{str(neuron_type)[:-2]}", xlim=((0, tTest)), ylim=((-1.2, 1.2)), yticks=((-1, 1)))
        axes2[i+1].set(ylabel=f"{str(neuron_type)[:-2]}", xlim=((0, tTest)), ylim=((-1.2, 1.2)), yticks=((-1, 1)))
        sns.despine(ax=axes1[i+1], bottom=True)
        sns.despine(ax=axes2[i+1], bottom=True)
    df = pd.concat([df for df in dfsAll], ignore_index=True)

    axes1[0].plot(times, tarX1, label='target', color='k')
    axes2[0].plot(times, tarX2, label='target', color='k')
    axes1[0].set(ylabel=r"$\mathbf{x}_1(t)$", xlim=((0, tTest)), ylim=((-1.2, 1.2)), yticks=((-1, 1)))
    axes2[0].set(ylabel=r"$\mathbf{x}_2(t)$", xlim=((0, tTest)), ylim=((-1.2, 1.2)), yticks=((-1, 1)))
    axes1[-1].set(xlabel='time (s)', xticks=((0, tTest)))
    axes2[-1].set(xlabel='time (s)', xticks=((0, tTest)))
    sns.despine(ax=axes1[0], bottom=True)
    sns.despine(ax=axes2[0], bottom=True)
    sns.despine(ax=axes1[-1], bottom=False)
    sns.despine(ax=axes2[-1], bottom=False)
    plt.tight_layout()
    fig1.savefig('plots/figures/multiply_ens1.pdf')
    fig1.savefig('plots/figures/multiply_ens1.svg')
    fig2.savefig('plots/figures/multiply_ens2.pdf')
    fig2.savefig('plots/figures/multiply_ens2.svg')

    # fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=((6, 3)), sharex=True)
    # sns.barplot(data=df, x='neuron_type', y='error1', ax=ax)
    # sns.barplot(data=df, x='neuron_type', y='error2', ax=ax2)
    # ax.set(xlabel='', ylabel='Error')
    # ax2.set(xlabel='', ylabel='Error')
    # sns.despine(ax=ax)
    # sns.despine(ax=ax2)
    # fig.savefig('plots/figures/multiply_barplot.pdf')
    # fig.savefig('plots/figures/multiply_barplot.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((6, 1)), sharex=True)
    sns.barplot(data=df, x='neuron_type', y='error2', ax=ax)
    ax.set(xlabel='', ylabel='Error')
    sns.despine(ax=ax)
    fig.savefig('plots/figures/multiply_barplot.pdf')
    fig.savefig('plots/figures/multiply_barplot.svg')

# compare([LIF()], load=[])
compare([LIF(), Izhikevich(), Wilson(), Pyramidal()], load=[])