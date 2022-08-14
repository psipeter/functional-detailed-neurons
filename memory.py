import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp
from nengolib.synapses import ss2sim
from nengolib.signal import LinearSystem, s

from neuron_types import LIF, Izhikevich, Wilson, NEURON, nrnReset
from utils import LearningNode, trainDF, fitSinusoid
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
palette = sns.color_palette('dark')
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

def makeSignalCircle(t, dt=0.001, radius=1, rms=0.1, seed=0):
    phase = np.random.RandomState(seed=seed).uniform(0, 1)
    stim = nengo.processes.WhiteSignal(period=t, high=4, rms=rms, seed=seed)
    stim2 = nengo.processes.WhiteSignal(period=t, high=4, rms=rms, seed=50+seed)
    with nengo.Network() as model:
        inpt = nengo.Node(stim)
        inpt2 = nengo.Node(stim2)
        probe = nengo.Probe(inpt, synapse=None)
        probe2 = nengo.Probe(inpt2, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    stim_func = lambda tt: radius*np.sin(2*np.pi*(tt/t+phase)) + sim.data[probe][int(tt/dt)]
    stim_func2 = lambda tt: radius*np.cos(2*np.pi*(tt/t+phase)) + sim.data[probe2][int(tt/dt)]
    return stim_func, stim_func2

def makeTest(t, angle, radius=1, tGate=1):
    stim_func1 = lambda t: np.array([radius*np.sin(angle), radius*np.cos(angle)])
    stim_func2 = lambda t: 0 if t<tGate else 1
    return stim_func1, stim_func2

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=10,
    m=Uniform(20, 40), i=Uniform(-0.8, 0.8), stim_func1=lambda t: 0, stim_func2=lambda t: 0,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    dB=None, eB=None, wB=None, d0=None, e0=None, w0=None, d1=None, e1=None, w1=None,
    learn0=False, check0=False, learn1=False,
    eRate=1e-6, dRate=3e-6):

    with nengo.Network() as model:
        inpt1 = nengo.Node(stim_func1)
        inpt2 = nengo.Node(stim_func2)
        inpt = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        const = nengo.Node([1,1])
        pre = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        bias = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
        nengo.Connection(inpt1, inpt[0], synapse=None)
        nengo.Connection(inpt2, inpt[1], synapse=None)
        nengo.Connection(inpt, pre, synapse=None)

        if learn0:
            tarA = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)
            nengo.Connection(inpt, tarA, synapse=fTarget, seed=seed)
            nengo.Connection(bias, tarA, synapse=fTarget, seed=seed)
            connB = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            nodeB = LearningNode(bias, ens, 2, conn=connB, d=dB, e=eB, w=wB, eRate=eRate, dRate=dRate)
            nengo.Connection(bias.neurons, nodeB[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeB[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeB[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(const, nodeB[-2:], synapse=fTarget)
            nengo.Connection(nodeB, ens.neurons, synapse=None)
            conn0 = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            node0 = LearningNode(pre, ens, 2, conn=conn0, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, node0[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, node0[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, node0[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, node0[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(node0, ens.neurons, synapse=None)
            pTarA = nengo.Probe(tarA.neurons, synapse=None)  
        if check0:
            nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(wB, weights=True))
            nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(w0, weights=True))
        if learn1:
            pre2 = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
            ens2 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)  # feedforward
            ens3 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)  # target
            nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens2, synapse=fTarget, solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens3, synapse=fTarget, solver=NoSolver(wB, weights=True))
            nengo.Connection(inpt, pre2, synapse=fTarget)
            nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(w0, weights=True))
            nengo.Connection(pre2, ens3, synapse=fTarget, solver=NoSolver(w0, weights=True))
            conn1 = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            node1 = LearningNode(ens, ens2, 2, conn=conn1, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, node1[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, node1[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(ens3.neurons, node1[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(node1, ens2.neurons, synapse=None)
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)     

        pInpt = nengo.Probe(inpt, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, NEURON): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, NEURON): nrnReset(sim, model)

    if learn0:
        dB, eB, wB = nodeB.d, nodeB.e, nodeB.w    
        d0, e0, w0 = node0.d, node0.e, node0.w
    if learn1:
        e1, w1 = node1.e, node1.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        ens=sim.data[pEns],
        tarA=sim.data[pTarA] if learn0 else None,
        ens2=sim.data[pEns2] if learn1 else None,
        ens3=sim.data[pEns3] if learn1 else None,
        eB=eB,
        dB=dB,
        wB=wB,
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
    )

def goTest(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=100,
    m=Uniform(20, 40), stim_func1=lambda t: 0, stim_func2=lambda t: 0,
    fTarget=DoubleExp(1e-3, 1e-1), f1=DoubleExp(1e-3, 1e-1), tGate=1,
    w0=None, w1=None, wB=None):

    wInh = -3e-4*np.ones((nPre, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func1)
        gate = nengo.Node(stim_func2)
        bias = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        pre = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        inh = nengo.Ensemble(nPre, 1, max_rates=m, intercepts=Uniform(0.5, 1), seed=seed)
        diff = nengo.Ensemble(nEns, 2, max_rates=m, neuron_type=neuron_type, seed=seed)
        ens = nengo.Ensemble(nEns, 2, max_rates=m, neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(gate, inh, synapse=None)
        nengo.Connection(pre, diff, synapse=fTarget, solver=NoSolver(w0, weights=True))
        nengo.Connection(inh.neurons, diff.neurons, synapse=fTarget, transform=wInh.T)
        nengo.Connection(diff, ens, synapse=f1, solver=NoSolver(w1, weights=True))
        nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(w1, weights=True))
        nengo.Connection(ens, diff, synapse=f1, solver=NoSolver(-w1, weights=True))
        nengo.Connection(bias, diff, synapse=fTarget, solver=NoSolver(wB, weights=True))
        nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(wB, weights=True))

        pInpt = nengo.Probe(inpt, synapse=None)
        pGate = nengo.Probe(gate, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pDiff = nengo.Probe(diff.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, NEURON): neuron.h.init()
        sim.run(tGate+t, progress_bar=True)
        if isinstance(neuron_type, NEURON): nrnReset(sim, model)

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        gate=sim.data[pGate],
        pre=sim.data[pPre],
        diff=sim.data[pDiff],
        ens=sim.data[pEns],
    )


def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, tGate=1,
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), load=[]):

    print(f'Neuron type: {neuron_type}')

    if 0 in load:
        data = np.load(f"data/memory_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
        dB, eB, wB = data['dB'], data['eB'], data['wB']
    else:
        print('train d0, e0, w0 from pre to diff')
        d0, e0, w0 = None, None, None
        dB, eB, wB = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=n)
            data = go(neuron_type, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth, stim_func1=stim_func1, stim_func2=stim_func2)
            dB, eB, wB = data['dB'], data['eB'], data['wB']
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/memory_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "memory", neuron_type, "pre", n, nTrain)
        stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=0)
        data = go(neuron_type, learn0=True, eRate=0,
            nEns=nEns, t=tTrain, dt=dt,
            dB=dB, eB=eB, wB=wB,
            d0=d0, e0=e0, w0=w0,
            fTarget=fTarget, fSmooth=fSmooth, stim_func1=stim_func1, stim_func2=stim_func2)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "memory", neuron_type, "pre", -1, 0)

    if 1 in load:
        data = np.load(f"data/memory_{neuron_type}.npz")
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        d2 = -d1
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('train d1 and f1 for diff/ens to compute identity')
        targets = np.zeros((nTrain, int(tTrain/dt), 2))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt, check0=True,
                wB=wB,
                w0=w0,
                fTarget=fTarget, fSmooth=fSmooth, stim_func1=stim_func1, stim_func2=stim_func2)
            targets[n] = fTarget.filt(fTarget.filt(data['inpt'], dt=dt), dt=dt)
            spikes[n] = data['ens']
        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrain, dt=dt, network="memory", neuron_type=neuron_type, ens="ens")
        d2 = -d1
        f1 = DoubleExp(tauRise1, tauFall1)
        np.savez(f"data/memory_{neuron_type}.npz",
            dB=dB, eB=eB, wB=wB,
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)
        times = data['times']
        inpt = data['inpt']
        target = fTarget.filt(fTarget.filt(data['inpt'], dt=dt), dt=dt)
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        fig, ax = plt.subplots()
        ax.plot(times, target, label='target')
        ax.plot(times, xhat, label='xhat')
        ax.legend()
        ax.set(yticks=((-1,1)))
        fig.savefig(f'plots/memory/decode_{neuron_type}.pdf')

    if 2 in load:
        data = np.load(f"data/memory_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from diff/ens to ens')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=n)
            data = go(neuron_type, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                wB=wB,
                w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                fTarget=fTarget, fSmooth=fSmooth, stim_func1=stim_func1, stim_func2=stim_func2)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/memory_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
                "memory", neuron_type, "ens", n, nTrain)
        stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=0)
        data = go(neuron_type, learn1=True, eRate=0,
            nEns=nEns, t=tTrain, dt=dt,
            wB=wB,
            w0=w0,
            d1=d1, e1=e1, w1=w1, f1=f1,
            fTarget=fTarget, fSmooth=fSmooth, stim_func1=stim_func1, stim_func2=stim_func2)
        plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
            "memory", neuron_type, "ens", -1, 0)

    dfs = []
    columns = ('neuron_type', 'trial', 't', 'xhat0', 'xhat1', 'target0', 'target1', 'error')
    print('estimating error')
    angles = np.linspace(0, 2*np.pi, nTest+1)
    for n in range(nTest):
        stim_func1, stim_func2 = makeTest(tTest, angle=angles[n], tGate=tGate)
        data = goTest(neuron_type,
            nEns=nEns, t=tTest, tGate=tGate, dt=dt,
            wB=wB,
            w0=w0,
            w1=w1, f1=f1,
            fTarget=fTarget,
            stim_func1=stim_func1, stim_func2=stim_func2)
        times = data['times']
        target = data['inpt']
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        for idx, t in enumerate(times):
            error = rmse(xhat[idx], target[idx]) if t>tGate else 0
            dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, t, xhat[idx,0], xhat[idx,1], target[idx,0], target[idx,1], error]], columns=columns))
        
        fig, ax, = plt.subplots()
        ax.plot(times, target, label='target')
        ax.plot(times, xhat, label='xhat')
        ax.legend()
        ax.set(yticks=((-1,1)), xticks=((0, tGate, tGate+tTest)))
        fig.savefig(f'plots/memory/test_{neuron_type}_{n}.pdf')

        fig, ax = plt.subplots()
        ax.plot(xhat[:,0], xhat[:,1], label='xhat, rmse=%.3f'%error, zorder=1)
        ax.scatter(target[0,0], target[0,1], s=10, color='k', label='target', zorder=2)
        ax.legend()
        ax.set(xlabel=r"$\mathbf{x}_0$", ylabel=r"$\mathbf{x}_1$", yticks=((-1,1)), xticks=((-1,1)))
        fig.savefig(f'plots/memory/space_{neuron_type}_{n}.pdf')
        plt.close('all')

    return dfs


def compare(neuron_types, eRates=[1e-6, 3e-6, 3e-7, 1e-7], nTrain=10, tTrain=10, nTest=10, tTest=10, load=[], replot=False):

    if not replot:
        dfs = []
        for i, neuron_type in enumerate(neuron_types):
            df = run(neuron_type, nTrain, nTest, tTrain, tTest, eRate=eRates[i], load=load)
            dfs.extend(df)
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/memory.pkl")
    else:
        data = pd.read_pickle(f"data/memory.pkl")
    print(data)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=((5.2, 2)), gridspec_kw={'width_ratios': [1,2]})
    target0 = data.query("trial==0 & neuron_type=='LIF' & t==0.001")['target0'].to_numpy()[0]
    target1 = data.query("trial==0 & neuron_type=='LIF' & t==0.001")['target1'].to_numpy()[0]
    times = np.arange(0, 2*np.pi, 0.001)
    axes[0].plot(np.sin(times), np.cos(times), color='gray', alpha=0.5, linewidth=0.5, linestyle='--', zorder=1)
    axes[0].scatter(target0, target1, s=10, color='k', alpha=0.5, zorder=2)
    for i, neuron_type in enumerate(neuron_types):
        nt = str(neuron_type)[:-2]
        xhat0 = data.query("trial==0 & neuron_type==@nt")['xhat0'].to_numpy()
        xhat1 = data.query("trial==0 & neuron_type==@nt")['xhat1'].to_numpy()
        axes[0].plot(xhat0, xhat1, color=palette[i], linewidth=0.5)
    sns.barplot(data=data.query("t>1"), x='neuron_type', y='error', ax=axes[1])
    axes[0].set(xticks=((-1,1)), yticks=((-1, 1)), xlim=((-1.2, 1.2)), ylim=((-1.2, 1.2)), xlabel=r"$\mathbf{x}_0$", ylabel=r"$\mathbf{x}_1$")
    axes[1].set(xlabel='', ylim=((0, 0.3)), yticks=((0, 0.3)), ylabel='Error')
    plt.tight_layout()
    fig.savefig('plots/figures/memory_combined_v2.svg')

compare([LIF(), Izhikevich(), Wilson(), NEURON('Pyramidal')], load=[0,1,2], replot=True)

def print_time_constants():
    for neuron_type in ['LIF()', 'Izhikevich()', 'Wilson()', 'Pyramidal()']:
        data = np.load(f"data/memory_{neuron_type}.npz")
        rise, fall = 1000*data['tauRise1'], 1000*data['tauFall1']
        print(f"{neuron_type}:  \t rise {rise:.3}, fall {fall:.4}")
# print_time_constants()
