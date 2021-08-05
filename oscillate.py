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
from nengolib.stats import sphere, ball

from neuron_types import LIF, Izhikevich, Wilson, Pyramidal, nrnReset
from utils import LearningNode, trainDF, trainD, fitSinusoid, getGainLIF
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})


def go(neuron_type, t=10, seed=0, dt=1e-3, nEns=100, w=2*np.pi, max_rates=Uniform(30, 60),
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1),
    d0=None, e0=None, w0=None, d1=None, learn0=False, learn1=False, test=False,
    eRate=1e-6, dRate=3e-6, tKick=0.1):

    tauFall = (-1.0 / np.array(fTarget.poles))[1]
    def feedback(x):
        r = np.maximum(np.sqrt(x[0]**2 + x[1]**2), 1e-9)
        dx0 = x[0]*(1-r**2)/r - x[1]*w 
        dx1 = x[1]*(1-r**2)/r + x[0]*w 
        return [tauFall*dx0 + x[0],  tauFall*dx1 + x[1]]

    d0 = d0 if np.any(d0) else np.zeros((nEns, 2))
    d1 = d1 if np.any(d1) else d0
    gain, bias = getGainLIF(nEns, max_rates)

    with nengo.Network() as model:
        inpt = nengo.Node(lambda t: [1, 0] if t<tKick else [0,0])  # square wave kick
        tarA = nengo.Ensemble(nEns, 2, neuron_type=nengo.LIF(), gain=gain, bias=bias, seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
        xhatTarA = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        xhatEns = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())

        nengo.Connection(inpt, tarA, synapse=fTarget)
        connTarFB = nengo.Connection(tarA, tarA, synapse=fTarget, function=feedback)
        connTarOut = nengo.Connection(tarA, xhatTarA, synapse=fTarget, function=feedback)
        connEnsOut = nengo.Connection(ens, xhatEns, synapse=fTarget, solver=NoSolver(d1, weights=False))  # readout decode

        if learn1:
            connSupv = nengo.Connection(tarA, ens, synapse=fTarget, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            nodeSupv = LearningNode(tarA, ens, 2, conn=connSupv, d=d0, e=e0, w=w0, eRate=eRate, dRate=0)
            nengo.Connection(tarA.neurons, nodeSupv[:nEns], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeSupv[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(xhatTarA, nodeSupv[-2:], synapse=None)
            nengo.Connection(nodeSupv, ens.neurons, synapse=None)

        if test:
            off = nengo.Node(lambda t: 0 if t<=tKick else -1e4)
            connSupv = nengo.Connection(tarA, ens, synapse=fTarget, solver=NoSolver(w0, weights=True))  # kick
            connEnsFB = nengo.Connection(ens, ens, synapse=fTarget, solver=NoSolver(w0, weights=True))  # recurrent
            nengo.Connection(off, tarA.neurons, synapse=None, transform=np.ones((nEns, 1)))  # remove kick

        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pXhatTarA = nengo.Probe(xhatTarA, synapse=None)
        pXhatEns = nengo.Probe(xhatEns, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Pyramidal): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)

    if learn0:
        d0 = sim.data[connTarFB].weights
    if learn1:
        e0, w0 = nodeSupv.e, nodeSupv.w

    return dict(
        times=sim.trange(),
        ens=sim.data[pEns],
        tarA=sim.data[pTarA],
        xhatTarA=sim.data[pXhatTarA],
        xhatEns=sim.data[pXhatEns],
        e0=e0,
        d0=d0,
        w0=w0,
    )

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, tTransTest=0,
    nEns=500, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1),
    load=[], tKick=0.1):

    rng = np.random.RandomState(seed=0)
    if 0 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        d0 = data['d0']
    else:
        print('decoders for tarA')
        data = go(neuron_type, nEns=nEns, t=tTrain, fTarget=fTarget, learn0=True, tKick=tKick)
        d0 = data['d0'].T
        np.savez(f"data/oscillate_{neuron_type}.npz",
            d0=d0)

        fig, ax = plt.subplots()
        ax.plot(data['times'], data['xhatTarA'], label='xhat (ReLU)')
        ax.plot(data['times'], data['xhatEns'], label='xhat (ens)')
        ax.legend()
        fig.savefig('plots/oscillate/tarA.pdf')

    if 1 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        e0, w0 = data['e0'], data['w0']
    else:
        print('encoders/weights for supv')
        e0, w0 = None, None
        for n in range(nTrain):
            data = go(neuron_type, learn1=True, eRate=eRate, tKick=rng.uniform(tKick/2, tKick*2),
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0,
                e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            e0, w0 = data['e0'], data['w0']
            np.savez(f"data/oscillate_{neuron_type}.npz",
                d0=d0,
                e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillate", neuron_type, "ens", n, nTrain)

        print('checking activities/estimate with zero learning rate')
        data = go(neuron_type, learn1=True, eRate=0, tKick=tKick,
            nEns=nEns, t=tTrain, dt=dt,
            d0=d0,
            e0=e0, w0=w0,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "oscillate", neuron_type, "ens", -1, 0)

        fig, ax = plt.subplots()
        ax.plot(data['times'], data['xhatTarA'], label='xhat (ReLU)')
        ax.plot(data['times'], data['xhatEns'], label='xhat (ens)')
        ax.legend()
        fig.savefig(f'plots/oscillate/supervised_{neuron_type}.pdf')

    if 2 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        d1 = data['d1']
    else:
        targets = np.zeros((1, int(tTest/dt), 2))
        spikes = np.zeros((1, int(tTest/dt), nEns))
        print('decoders for ens (correct magnitude errors)')
        data = go(neuron_type, test=True, tKick=3*tKick,
            nEns=nEns, t=tTest, dt=dt,
            d0=d0,
            w0=w0,
            fTarget=fTarget)
        times = data['times']
        xhat = data['xhatEns']
        spikes[0] = data['ens']
        # fit the frequency, phase, and magnitude of a sinusoid to the current estimate xhat
        errorRMSE0, errorFreq0, freq0, phase0, mag0, base0 = fitSinusoid(xhat[:,0], neuron_type, tTrans=tTransTest, dt=dt)
        errorRMSE1, errorFreq1, freq1, phase1, mag1, base1 = fitSinusoid(xhat[:,1], neuron_type, tTrans=tTransTest, dt=dt)
        sin0 = np.sin(freq0*(times+phase0)).reshape(-1, 1)
        sin1 = np.sin(freq1*(times+phase1)).reshape(-1, 1)
        print(np.max(sin0), np.max(sin1), np.max(xhat))
        # compute decoders that translate between current spikes and a sinusoid with the above frequency and phase, but magnitude 1
        tarX = np.concatenate((sin0, sin1), axis=1)
        print(tarX.shape)
        targets[0] = tarX
        d1 = trainD(spikes, targets, 1, fTarget, dt=dt)
        np.savez(f"data/oscillate_{neuron_type}.npz",
            d0=d0,
            e0=e0, w0=w0,
            d1=d1)
        xhat2 = np.dot(fTarget.filt(data['ens'], dt=dt), d1)
        fig, ax = plt.subplots()
        ax.plot(data['times'], tarX, label=f'target (w={freq0:.2f}, {freq1:.2f})')
        ax.plot(data['times'], xhat2, label='xhat')
        ax.set(xlim=((0, tTest)), xticks=((0, tTest)), yticks=((-1, 1)), ylim=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        ax.legend()
        fig.savefig(f'plots/oscillate/decode_{neuron_type}.pdf')

    dfs = []
    columns = ('neuron_type', 'n', 'error rmse', 'error freq', 'dimension')
    print('estimating error')
    for n in range(nTest):
        data = go(neuron_type, test=True, tKick=rng.uniform(tKick*2, tKick*4),
            nEns=nEns, t=tTest, dt=dt,
            d0=d0,
            w0=w0,
            d1=d1,
            fTarget=fTarget)
        times = data['times']
        xhat = data['xhatEns']
        errorRMSE0, errorFreq0, freq0, phase0, mag0, base0 = fitSinusoid(xhat[:,0], neuron_type, tTrans=tTransTest, dt=dt, mag=False)
        errorRMSE1, errorFreq1, freq1, phase1, mag1, base1 = fitSinusoid(xhat[:,1], neuron_type, tTrans=tTransTest, dt=dt, mag=False)
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE0, errorFreq0, '0']], columns=columns))
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE1, errorFreq1, '1']], columns=columns))
        sin0 = np.sin(freq0*(times+phase0))
        sin1 = np.sin(freq1*(times+phase1))
    return times, xhat, sin0, sin1, freq0, freq1, dfs

def compare(neuron_types, nTrain=10, tTrain=20, nTest=1, tTest=20, tTransTest=10, load=[], tKick=0.1,
    eRates=[3e-7, 1e-6, 1e-7, 3e-8]):

    dfsAll = []
    fig, ax = plt.subplots(figsize=((5.25, 1.5)))
    fig2, ax2 = plt.subplots(figsize=((5.25, 1.5)))
    for i, neuron_type in enumerate(neuron_types):
        times, xhat, sin0, sin1, freq0, freq1, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest,
            eRate=eRates[i], tTransTest=tTransTest, tKick=tKick, load=load)
        dfsAll.extend(dfs)
        ax.plot(times, xhat[:,0], label=f"{str(neuron_type)[:-2]}", linewidth=0.5)
        ax2.plot(times, xhat[:,1], label=f"{str(neuron_type)[:-2]}", linewidth=0.5)
    df = pd.concat([df for df in dfsAll], ignore_index=True)

    ax.plot(times, np.sin(2*np.pi*times), label=r'sin$(2\pi t)$', color='k', linewidth=0.5)
    ax.set(xlim=((0, tTest)), xticks=(()), yticks=((-1, 1)), ylim=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_dim1.pdf')
    fig.savefig('plots/figures/oscillate_dim1.svg')

    ax2.plot(times, np.sin(2*np.pi*times), label=r'sin$(2\pi t)$', color='k', linewidth=0.5)
    ax2.set(xlim=((0, tTest)), xticks=(()), yticks=((-1, 1)), ylim=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax2.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig2.savefig('plots/figures/oscillate_dim2.pdf')
    fig2.savefig('plots/figures/oscillate_dim2.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='neuron_type', y='error rmse', ax=ax)
    ax.set(xlabel='', ylim=((0, 0.1)), yticks=((0, 0.1)), ylabel='Error (RMSE)')
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_barplot_rmse.pdf')
    fig.savefig('plots/figures/oscillate_barplot_rmse.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='neuron_type', y='error freq', ax=ax)
    ax.set(xlabel='', ylim=((0, 0.1)), yticks=((0, 0.1)), ylabel=r'Error ($\omega$)')
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_barplot_freq.pdf')
    fig.savefig('plots/figures/oscillate_barplot_freq.svg')


compare([Wilson()], nTrain=10, eRates=[3e-8], load=[])