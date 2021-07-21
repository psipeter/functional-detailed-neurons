import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp
from nengolib.synapses import ss2sim
from nengolib.signal import LinearSystem, cont2discrete

from neuron_types import LIF, Izhikevich, Wilson, Pyramidal, nrnReset
from utils import LearningNode, trainDF, fitSinusoid
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})


def makeSignal(t, phase, dt, tauRise=1e-3, tauFall=1e-1):
    # account for finite dt and tau when calculating the A matrix necessary on the recurrent transform
    # see https://forum.nengo.ai/t/oscillator-example/513/2)
    fTarget = (cont2discrete(Lowpass(tauRise), dt=dt) * cont2discrete(Lowpass(tauFall), dt=dt))
    idealA= [[0, 2*np.pi], [-2*np.pi, 0]]
    dsys = cont2discrete(LinearSystem((idealA, [[1], [0]], [[1, 0]], [[0]])), dt=dt)
    simA = ss2sim(dsys, fTarget, dt=None).ss[0]
    stim_func = lambda t: [0.1/dt, 0] if t<=dt else [0,0]
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        tarX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        nengo.Connection(inpt, tarX, synapse=fTarget)  # kick
        nengo.Connection(tarX, tarX, synapse=fTarget, transform=simA)  # recurrent
        pTarX = nengo.Probe(tarX, synapse=None, sample_every=dt)
    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        sim.run(t+1+dt, progress_bar=False)
    targets = sim.data[pTarX]
    # fig, ax = plt.subplots()
    # ax.plot(targets, color='k')
    # ax.set(ylim=((-1, 1)))
    # fig.savefig('plots/oscillate/communicate.pdf')
    # raise
    # return targets, simA
    return lambda t: targets[int((t+phase)/dt)], simA

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=10,
    m=Uniform(20, 40), stim_func=lambda t: [0,0], simA=[[0,1],[-1,0]], tKick=0.1,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, learn0=False, learn1=False, test=False,
    eRate=1e-6, dRate=3e-6):

    A = [[1, 0.1*2*np.pi], [-0.1*2*np.pi, 1]]
    weightsFF = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weightsSupv = w1 if (np.any(w1) and learn1) else np.zeros((nEns, nEns))
    weightsFB = w1 if (np.any(w1) and not learn0 and not learn1) else np.zeros((nEns, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        tarX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarX2 = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarA = nengo.Ensemble(nEns, 2, max_rates=m, neuron_type=ReLu(), seed=seed)
        tarA2 = nengo.Ensemble(nEns, 2, max_rates=m, neuron_type=ReLu(), seed=seed)
        pre = nengo.Ensemble(nPre, 2, max_rates=m, neuron_type=ReLu(), seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt, tarX, synapse=fTarget,)  # kick
        nengo.Connection(tarX, tarX2, synapse=None, transform=simA)
        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarA, synapse=fTarget)
        nengo.Connection(tarX2, tarA2, synapse=fTarget)
        connFF = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weightsFF, weights=True))  # kick
        # connFB = nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(weightsFB, weights=True))  # recurrent

        if learn0:  # learn to receive supervised "recurrent" input from ReLU
            nodeFF = LearningNode(pre, ens, 2, conn=connFF, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, nodeFF[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeFF[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeFF[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, nodeFF[-2:], synapse=fTarget)
            nengo.Connection(nodeFF, ens.neurons, synapse=None)

        if learn1: # learn to receive supervised "recurrent" input from neuron_type
            connSupv = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(weightsSupv, weights=True))
            nodeSupv = LearningNode(ens, ens2, 2, conn=connSupv, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, nodeSupv[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, nodeSupv[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA2.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(nodeSupv, ens2.neurons, synapse=None)

        if test:
            # conn2 = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(weightsFB, weights=True))  # recurrent
            connFB = nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(weightsFB, weights=True))  # recurrent
            off = nengo.Node(lambda t: 0 if t<=tKick else -1e4)
            nengo.Connection(off, pre.neurons, synapse=None, transform=np.ones((nPre, 1)))


        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarA2 = nengo.Probe(tarA2.neurons, synapse=None)
        pTarX = nengo.Probe(tarX, synapse=None)
        pTarX2 = nengo.Probe(tarX2, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Pyramidal): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)
    
    if learn0:
        d0, e0, w0 = nodeFF.d, nodeFF.e, nodeFF.w
    if learn1:
        e1, w1 = nodeSupv.e, nodeSupv.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens=sim.data[pEns],
        ens2=sim.data[pEns2],
        tarA=sim.data[pTarA],
        tarX=sim.data[pTarX],
        tarX2=sim.data[pTarX2],
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
        f1=f1,
    )

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, tTransTrain=0, tTransTest=0,
    nEns=100, dt=1e-3, tKick=0.1, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1),
    load=[]):

    print(f'Neuron type: {neuron_type}')

    if 0 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens (kick input)')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func, simA = makeSignal(tTrain, n/nTrain, dt)
            data = go(neuron_type, stim_func=stim_func, simA=simA, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/oscillate_{neuron_type}.npz", d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillate", neuron_type, "ens", n, nTrain)

    if 1 in load:
        d1, tauRise, tauFall = data['d1'], data['tauRise'], data['tauFall']
        f1 = DoubleExp(tauRise, tauFall)
    else:
        print('train d1 and f1 for ens to compute the A matrix for the oscillator')
        targets = np.zeros((nTrain, int(tTrain/dt), 2))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func, simA = makeSignal(tTrain, n/nTrain, dt)
            data = go(neuron_type, stim_func=stim_func, simA=simA, 
                nEns=nEns, t=tTrain+tTransTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            targets[n] = fTarget.filt(data['tarX2'], dt=dt)[int(tTransTrain/dt):]
            spikes[n] = data['ens'][int(tTransTrain/dt):]
            fig, ax = plt.subplots()
            ax.plot(data['times'], data['tarX'], color='k')
            fig.savefig(f'plots/oscillate/communicate_{n}.pdf')
        d1, tauRise, tauFall = trainDF(spikes, targets, nTrain, dt=dt, network="oscillate", neuron_type=neuron_type, ens="ens")
        f1 = DoubleExp(tauRise, tauFall)
        np.savez(f"data/oscillate_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise=tauRise, tauFall=tauFall)

    if 2 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from ens to ens2')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func, simA = makeSignal(tTrain, n/nTrain, dt)
            data = go(neuron_type, stim_func=stim_func, simA=simA, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, f1=f1,
                e1=e1, w1=w1,
                fTarget=fTarget, fSmooth=fSmooth)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/oscillate_{neuron_type}.npz",
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise=tauRise, tauFall=tauFall,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens'], dt=dt),
                "oscillate", neuron_type, "ens2", n, nTrain)

    # another readout step?

    dfs = []
    columns = ('neuron_type', 'n', 'error rmse', 'error freq', 'dimension')
    print('estimating error')
    for n in range(nTest):
        stim_func, simA = makeSignal(tTest+tTransTest, n/nTest, dt)
        data = go(neuron_type, stim_func=stim_func, simA=simA,  test=True,
            nEns=nEns, t=tTest+tTransTest, dt=dt, tKick=tKick,
            d0=d0, e0=e0, w0=w0,
            d1=d1, f1=f1,
            e1=e1, w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        times = data['times']
        tarX = fTarget.filt(data['tarX'], dt=dt)
        aEns = f1.filt(data['ens'], dt=dt)
        aEns2 = f1.filt(data['ens2'], dt=dt)
        xhat = np.dot(aEns, d1)
        xhat2 = np.dot(aEns2, d1)
        fig, ax = plt.subplots()
        ax.plot(times, tarX, color='k')
        ax.plot(times, xhat)
        ax.plot(times, xhat2)
        fig.savefig(f'plots/oscillate/communicateTest_{n}.pdf')
        # raise
        errorRMSE0, errorFreq0, freq0, phase0, mag0, base0 = fitSinusoid(xhat[int(tTransTest/dt):,0], neuron_type, dt=dt)  # muFreq=muFreq, sigmaFreq=sigmaFreq, base=base
        errorRMSE1, errorFreq1, freq1, phase1, mag1, base1 = fitSinusoid(xhat[int(tTransTest/dt):,1], neuron_type, dt=dt)  # muFreq=muFreq, sigmaFreq=sigmaFreq, base=base
        # error0 = rmse(xhat[int(tTrans/dt):,0], tarX[int(tTrans/dt):,0])
        # error1 = rmse(xhat[int(tTrans/dt):,1], tarX[int(tTrans/dt):,1])
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE0, errorFreq0, '0']], columns=columns))
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE1, errorFreq1, '1']], columns=columns))

    return times, tarX, xhat, [freq0, freq1], dfs

def compare(neuron_types, nTrain=10, tTrain=10, nTest=1, tTest=10, tTransTest=2, tKick=0.1, load=[],
    eRates=[3e-7, 1e-6, 1e-7, 3e-8], tTransTrains=[0, 5, 5, 10]):

    dfsAll = []
    fig, ax = plt.subplots(figsize=((5.25, 1.5)))
    fig2, ax2 = plt.subplots(figsize=((5.25, 1.5)))
    for i, neuron_type in enumerate(neuron_types):
        times, tarX, xhat, freq, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest,
            eRate=eRates[i], tTransTrain=tTransTrains[i], tTransTest=tTransTest, tKick=tKick, load=load)
        dfsAll.extend(dfs)
        ax.plot(times, xhat[:,0], label=f"{str(neuron_type)[:-2]}, " + r"$\omega=$" + f"{freq[0]:.2f}", linewidth=0.5)
        ax2.plot(times, xhat[:,1], label=f"{str(neuron_type)[:-2]}, " + r"$\omega=$" + f"{freq[1]:.2f}", linewidth=0.5)
    df = pd.concat([df for df in dfsAll], ignore_index=True)

    ax.plot(times, tarX[:,0], label='target, ' + r"$\omega=$" + f"{2*np.pi:.2f}", color='k', linewidth=0.5)
    # ax.axvline(tTransTest, color='k', linestyle=":", label="transient", linewidth=0.5)
    ax.set(xlim=((tTransTest, tTest+tTransTest)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_dim1.pdf')
    fig.savefig('plots/figures/oscillate_dim1.svg')

    ax2.plot(times, tarX[:,1], label='target, ' + r"$\omega=$" + f"{2*np.pi:.2f}", color='k', linewidth=0.5)
    # ax2.axvline(tTransTest, color='k', linestyle=":", label="transient", linewidth=0.5)
    ax2.set(xlim=((tTransTest, tTest+tTransTest)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax2.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig2.savefig('plots/figures/oscillate_dim2.pdf')
    fig2.savefig('plots/figures/oscillate_dim2.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='neuron_type', y='error rmse', ax=ax)
    ax.set(xlabel='', ylim=((0, 0.2)), yticks=((0, 0.2)), ylabel='Error (RMSE)')
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_barplot_rmse.pdf')
    fig.savefig('plots/figures/oscillate_barplot_rmse.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='neuron_type', y='error freq', ax=ax)
    ax.set(xlabel='', ylim=((0, 0.2)), yticks=((0, 0.2)), ylabel=r'Error ($\omega$)')
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_barplot_freq.pdf')
    fig.savefig('plots/figures/oscillate_barplot_freq.svg')


compare([LIF()], tTransTrains=[0], eRates=[3e-7], load=[])