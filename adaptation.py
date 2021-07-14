import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp

from utils import LearningNode, trainDF, trainD
from neuron_types import LIF, Izhikevich, Wilson, Pyramidal, nrnReset
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
        if seed%2==0: stim*=-1
    else:
        stim = u / np.min(u)
        if seed%2==0: stim*=-1
    stim_func = lambda t: stim[int(t/dt)]
    return stim_func

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=10,
    m=Uniform(20, 40), eRate=1e-6, dRate=3e-6,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1),
    d=None, e=None, w=None, learn=False, stim_func=lambda t: 0):

    weights = w if (np.any(w) and not learn) else np.zeros((nPre, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        tarX = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarA = nengo.Ensemble(nEns, 1, max_rates=m, neuron_type=ReLu(), seed=seed)
        pre = nengo.Ensemble(nPre, 1, max_rates=m, seed=seed)
        ens = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarX, synapse=fTarget)
        nengo.Connection(pre, tarA, synapse=fTarget)
        conn = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weights, weights=True))

        if learn:
            node = LearningNode(pre, ens, 1, conn=conn, d=d, e=e, w=w, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, node[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, node[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, node[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, node[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(node, ens.neurons, synapse=None)

        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pV = nengo.Probe(ens.neurons, 'voltage', synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarX = nengo.Probe(tarX, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Pyramidal): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)
    
    if learn:
        d, e, w = node.d, node.e, node.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens=sim.data[pEns],
        voltage=sim.data[pV],
        tarA=sim.data[pTarA],
        tarX=sim.data[pTarX],
        e=e,
        d=d,
        w=w,
    )

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate,
    nEns=30, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), load=[]):

    print(f'Neuron type: {neuron_type}')
    if 1 in load:
        data = np.load(f"data/adaptation_{neuron_type}.npz")
        d, e, w = data['d'], data['e'], data['w']
    else:
        print('train d, e, w from pre to ens')
        d, e, w = None, None, None
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type, learn=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d=d, e=e, w=w,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            d, e, w = data['d'], data['e'], data['w']
            np.savez(f"data/adaptation_{neuron_type}.npz", d=d, e=e, w=w)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "adaptation", neuron_type, "ens1", n, nTrain)

    if 2 in load:
        dOutNoF, dOut, tauRiseOut, tauFallOut = data['dOutNoF'], data['dOut'], data['tauRiseOut'], data['tauFallOut']
        fOut = DoubleExp(tauRiseOut, tauFallOut)
    else:
        print('train readout decoders and filters')
        targets = np.zeros((nTrain, int(tTrain/dt), 1))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d=d, e=e, w=w,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            targets[n] = fTarget.filt(data['tarX'], dt=dt)
            spikes[n] = data['ens']

        dOut, tauRiseOut, tauFallOut = trainDF(spikes, targets, nTrain, dt=dt, network="adaptation", ens='ens1', neuron_type=neuron_type)
        fOut = DoubleExp(tauRiseOut, tauFallOut)
        dOutNoF = trainD(spikes, targets, nTrain, fTarget, dt=dt)
        np.savez(f"data/adaptation_{neuron_type}.npz",
            d=d, e=e, w=w,
            dOutNoF=dOutNoF, dOut=dOut, tauRiseOut=tauRiseOut, tauFallOut=tauFallOut)

    dfs = []
    columns = ('neuron_type', 'n', 'error', 'filter')
    print('estimating error')
    for n in range(nTest):
        stim_func = makeSignal(tTest, dt=dt, seed=100+n)
        data = go(neuron_type,
            nEns=nEns, t=tTest, dt=dt,
            d=d, e=e, w=w,
            fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
        times = data['times']
        tarX = fTarget.filt(data['tarX'], dt=dt)
        aEnsNoF = fTarget.filt(data['ens'], dt=dt)
        aEns = fOut.filt(data['ens'], dt=dt)
        xhatNoF = np.dot(aEnsNoF, dOutNoF)
        xhat = np.dot(aEns, dOut)
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, rmse(xhatNoF, tarX), "default"]], columns=columns))
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, rmse(xhat, tarX), "trained"]], columns=columns))

    return times, tarX, xhatNoF, xhat, dfs

    # print('estimating spike adaptation')
    # dfsISI = []
    # columnsISI = ('neuron_type', 'n', 'ISI variance')
    # stim_func = lambda t: 0
    # data = go(neuron_type,
    #     nEns=nEns, t=tTest, dt=dt,
    #     d=d, e=e, w=w,
    #     fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
    # for n in range(nEns):
    #     spike_times = np.where(data['ens'][:,n]>0)[0]
    #     if len(spike_times) > 3:
    #         ISI = [x - spike_times[i-1] for i, x in enumerate(spike_times)][1:]
    #         # ISIvar = np.std(ISI)
    #         ISIvar = (np.max(ISI) - np.min(ISI)) / np.max(ISI)
    #     else:
    #         ISIvar = np.nan
    #     dfsISI.append(pd.DataFrame([[str(neuron_type)[:-2], n, ISIvar]], columns=columnsISI))
    # return times, tarX, xhatNoF, xhat, dfs, dfsISI

def compare(neuron_types, eRates=[3e-7, 3e-6, 3e-7, 1e-7], nTrain=10, tTrain=10, nTest=10, tTest=10, load=[]):

    dfsAll = []
    # dfsAllISI = []
    columns = ('neuron_type', 'n', 'error', 'filter')
    # columnsISI = ('neuron_type', 'n', 'ISI variance')
    fig, axes = plt.subplots(nrows=len(neuron_types)+1, ncols=1, figsize=((6, len(neuron_types))), sharex=True)
    for i, neuron_type in enumerate(neuron_types):
        # times, tarX, xhatNoF, xhat, dfs, dfsISI = run(neuron_type, nTrain, nTest, tTrain, tTest, eRate=eRates[i], load=load)
        times, tarX, xhatNoF, xhat, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest, eRate=eRates[i], load=load)
        dfsAll.extend(dfs)
        # dfsAllISI.extend(dfsISI)
        axes[i+1].plot(times, xhatNoF, label="default")
        axes[i+1].plot(times, xhat, label="trained")
        axes[i+1].set(ylabel=f"{str(neuron_type)[:-2]}", xlim=((0, tTest)), ylim=((-1.2, 1.2)), yticks=((-1, 1)))
        sns.despine(ax=axes[i+1], bottom=True)
    df = pd.concat([df for df in dfsAll], ignore_index=True)
    # dfISI = pd.concat([df for df in dfsAllISI], ignore_index=True)

    axes[0].plot(times, tarX, label='target', color='k')
    axes[0].set(ylabel=r"$\mathbf{x}(t)$", xlim=((0, tTest)), ylim=((-1.2, 1.2)), yticks=((-1, 1)))
    axes[1].legend(loc='upper right', frameon=False)
    axes[-1].set(xlabel='time (s)', xticks=((0, tTest)))
    sns.despine(ax=axes[0], bottom=True)
    sns.despine(ax=axes[-1], bottom=True)
    plt.tight_layout()
    fig.savefig('plots/figures/adaptation_state.pdf')
    fig.savefig('plots/figures/adaptation_state.svg')

    fig, ax = plt.subplots(figsize=((6, 1)))
    sns.barplot(data=df, x='neuron_type', y='error', hue='filter', ax=ax)
    ax.set(xlabel='', ylabel='Error', ylim=((0, 0.2)), yticks=((0, 0.2)))
    # ax.legend(loc='upper right', frameon=False)
    sns.despine()
    fig.savefig('plots/figures/adaptation_barplot.pdf')
    fig.savefig('plots/figures/adaptation_barplot.svg')

    # fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=((6, 2)), sharex=True)
    # sns.barplot(data=df, x='neuron_type', y='error', hue='filter', ax=ax)
    # ax.set(xlabel='', ylabel='Error', ylim=((0, 0.2)), yticks=((0, 0.2)))
    # ax.legend(loc='upper right', frameon=False)
    # sns.barplot(data=dfISI, x='neuron_type', y='ISI variance', ax=ax2)
    # ax2.set(xlabel='', ylim=((0, 0.8)), yticks=((0, 0.8)))
    # sns.despine()
    # fig.savefig('plots/figures/adaptation_barplot.pdf')
    # fig.savefig('plots/figures/adaptation_barplot.svg')

compare([LIF(), Izhikevich(), Wilson(), Pyramidal()], load=[0,1,2,3])