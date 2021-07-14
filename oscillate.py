import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp

from neuron_types import LIF, Izhikevich, Wilson, Pyramidal, nrnReset
from utils import LearningNode, trainDF, fitSinusoid
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif')


def makeSignal(t, dt=0.001, value=1, freq=1, seed=0):
    phase = 2*np.pi*seed / 10
    stim_func = lambda t: [value*np.sin(2*np.pi*freq*t+phase), value*np.cos(2*np.pi*freq*t+phase)]
    A = [[1, 1e-1*2*np.pi*freq], [-1e-1*2*np.pi*freq, 1]]  # tau*A + I
    return stim_func, A


def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=10,
    m=Uniform(20, 40), stim_func=lambda t: 0, A=[[1, 2*np.pi/10], [-2*np.pi/10, 1]],
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, learn0=False, learn1=False, test=False,
    eRate=1e-6, dRate=3e-6):

    weights0 = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weights1 = w1 if (np.any(w1) and not learn0 and not learn1) else np.zeros((nEns, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        tarX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarA = nengo.Ensemble(nEns, 2, max_rates=m, neuron_type=ReLu(), radius=2, seed=seed)
        pre = nengo.Ensemble(nPre, 2, max_rates=m, neuron_type=ReLu(), radius=2, seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarX, synapse=fTarget, transform=A)
        nengo.Connection(inpt, tarA, synapse=fTarget)
        conn0 = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weights0, weights=True))
        conn1 = nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(weights1, weights=True))

        if learn0:  # learn to receive the kick
            node0 = LearningNode(pre, ens, 2, conn=conn0, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, node0[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, node0[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, node0[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, node0[-2:], synapse=None)
            nengo.Connection(node0, ens.neurons, synapse=None)
        if learn1:  # learn the recurrent dynamics using a feedforward connection
            weights2 = w1 if np.any(w1) else np.zeros((nEns, nEns))
            tarA2 = nengo.Ensemble(nEns, 2, max_rates=m, neuron_type=ReLu(), radius=2, seed=seed)
            ens2 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
            nengo.Connection(tarX, tarA2, synapse=fTarget)
            conn2 = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(weights2, weights=True))
            node1 = LearningNode(ens, ens2, 2, conn=conn2, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, node1[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, node1[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA2.neurons, node1[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(node1, ens2.neurons, synapse=None)
            pTarA2 = nengo.Probe(tarA.neurons, synapse=None)
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        if test:  # send a kick through pre, then inhibit pre
            off = nengo.Node(lambda t: 1 if t>tDrive else 0)
            nengo.Connection(off, pre.neurons, synapse=None, transform=-1e4*np.ones((NPre, 1)))            

        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarX = nengo.Probe(tarX, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Pyramidal): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)
    
    if learn0:
        d0, e0, w0 = node0.d, node0.e, node0.w
    if learn1:
        e1, w1 = node1.e, node1.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens=sim.data[pEns],
        ens2=sim.data[pEns2] if learn1 else None,
        tarA=sim.data[pTarA],
        tarA2=sim.data[pTarA2] if learn1 else None,
        tarX=sim.data[pTarX],
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
    )

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate,
    freq=1, muFreq=1.0, sigmaFreq=0.1, tDrive=0.2, tTrans=2, base=False, evals=200,
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), load=[]):

    print(f'Neuron type: {neuron_type}')
    if 0 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func, A = makeSignal(tTrain, value=1.2, freq=freq, dt=dt, seed=n)
            data = go(neuron_type, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth,
                stim_func=stim_func, A=A)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/oscillate_{neuron_type}.npz", d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillate", neuron_type, "ens", n, nTrain)

    if 1 in load:
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('train d1 and f1 for ens')
        targets = np.zeros((nTrain, int((tTrain-tTrans)/dt), 2))
        spikes = np.zeros((nTrain, int((tTrain-tTrans)/dt), nEns))
        for n in range(nTrain):
            stim_func, A = makeSignal(tTrain, value=1.2, freq=freq, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth,
                stim_func=stim_func, A=A)
            targets[n] = fTarget.filt(data['tarX'], dt=dt)[int(tTrans/dt):]
            spikes[n] = data['ens'][int(tTrans/dt):]

        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrain, dt=dt, network="oscillate", neuron_type=neuron_type, ens="ens")
        f1 = DoubleExp(tauRise1, tauFall1)
        np.savez(f"data/oscillate_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)

    if 2 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from ens to ens2')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func, A = makeSignal(tTrain, value=1.2, freq=freq, dt=dt, seed=n)
            data = go(neuron_type, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                fTarget=fTarget, fSmooth=fSmooth,
                stim_func=stim_func, A=A)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/oscillate_{neuron_type}.npz",
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
                "oscillate", neuron_type, "ens2", n, nTrain)

    dfs = []
    columns = ('neuron_type', 'n', 'error', 'dim')
    print('estimating error')
    for n in range(nTest):
        stim_func, A = makeSignal(tTrain, value=1.2, freq=freq, dt=dt, seed=n+0.5)
        data = go(neuron_type,
            nEns=nEns, t=tTest, dt=dt,
            d0=d0, e0=e0, w0=w0,
            d1=d1, e1=e1, w1=w1, f1=f1,
            fTarget=fTarget, fSmooth=fSmooth,
            stim_func=stim_func, A=A)

        times = data['times']
        tarX = fTarget.filt(data['tarX'], dt=dt)
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)

        freq0, phase0, mag0, base0 = fitSinusoid(times, xhat[:,0], int(tTrans/dt), muFreq=muFreq, sigmaFreq=sigmaFreq, evals=evals, base=base)
        freq1, phase1, mag1, base1 = fitSinusoid(times, xhat[:,1], int(tTrans/dt), muFreq=muFreq, sigmaFreq=sigmaFreq, evals=evals, base=base)
        tarFit0 = base0+mag0*np.sin(times*2*np.pi*freq0+phase0)
        tarFit1 = base1+mag1*np.sin(times*2*np.pi*freq1+phase1)
        rmseError0 = rmse(xhat[int(tTrans/dt):,0], tarFit0[int(tTrans/dt):])
        rmseError1 = rmse(xhat[int(tTrans/dt):,1], tarFit1[int(tTrans/dt):])
        freqError0 = np.abs(freq-freq0)
        freqError1 = np.abs(freq-freq1)
        # error0 = (1+freqError0) * rmseError0
        # error1 = (1+freqError1) * rmseError1
        error0 = rmseError0
        error1 = rmseError1
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, error0, '0']], columns=columns))
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, error1, '1']], columns=columns))

    return times, xhat, tarFit0, tarFit1, dfs

def compare(neuron_types, eRates=[3e-7, 3e-6, 3e-7, 1e-7], nTrain=10, tTrain=10, nTest=3, tTest=10, tTrans=2, load=[]):

    dfsAll = []
    columns = ('neuron_type', 'n', 'error1', 'error2')
    fig, axes = plt.subplots(nrows=len(neuron_types), ncols=1, figsize=((6, len(neuron_types))), sharex=True)
    for i, neuron_type in enumerate(neuron_types):
        times, xhat, tarFit0, tarFit1, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest,
            eRate=eRates[i], tTrans=tTrans, load=load)
        dfsAll.extend(dfs)
        axes[i].plot(times, tarFit0, label=r'$\mathbf{x}_0(t)$', color='k', linewidth=0.5)
        axes[i].plot(times, tarFit1, label=r'$\mathbf{x}_1(t)$', color='k', linestyle='--', linewidth=0.5)
        axes[i].plot(times, xhat[:,0], label=r'$\mathbf{\hat{x}}_0(t)$')
        axes[i].plot(times, xhat[:,1], label=r'$\mathbf{\hat{x}}_1(t)$')
        axes[i].set(ylabel=f"{str(neuron_type)[:-2]}", xlim=((0, tTest)), ylim=((-1.2, 1.2)), yticks=((-1, 1)))
        sns.despine(ax=axes[i], bottom=True)
    axes[0].legend(loc='upper right', frameon=False)
    df = pd.concat([df for df in dfsAll], ignore_index=True)
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_states.pdf')
    fig.savefig('plots/figures/oscillate_states.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((6, 1)), sharex=True)
    sns.barplot(data=df, x='neuron_type', y='error', hue='dim', ax=ax)
    ax.set(xlabel='', ylabel='Error')
    sns.despine(ax=ax)
    fig.savefig('plots/figures/oscillate_barplot.pdf')
    fig.savefig('plots/figures/oscillate_barplot.svg')

compare([LIF(), Izhikevich()], load=[])
# compare([LIF(), Izhikevich(), Wilson(), Pyramidal()], load=[])