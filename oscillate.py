import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver, LstsqL2
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp
from nengolib.synapses import ss2sim
from nengolib.signal import LinearSystem, cont2discrete

from neuron_types import LIF, Izhikevich, Wilson, NEURON, nrnReset
from utils import LearningNode, trainDF, fitSinusoid
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
palette = sns.color_palette('colorblind')
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

def makeSignal(t, phase, w=2*np.pi, aEnv=0.7, aSin=0.7, dt=1e-3):
    idx = int(phase * t / dt)
    times = np.arange(0, 3*t, dt)[idx:]
    envelope = 1 + aEnv*np.sin(2*(2*np.pi/t)*times)
    sin = aSin*np.sin(w*times)
    cos = aSin*np.cos(w*times)
    stim_func1 = lambda t: envelope[int(t/dt)] * sin[int(t/dt)]
    stim_func2 = lambda t: envelope[int(t/dt)] * cos[int(t/dt)]
    return stim_func1, stim_func2

def makeKick(t, phase, tSupv=0.1, w=2*np.pi, dt=1e-3):
    idx = int(phase * t / dt)
    times = np.arange(0, 3*t, dt)[idx:]
    sin = np.sin(w*times)
    cos = np.cos(w*times)
    stim_func1 = lambda t: sin[int(t/dt)] if t<tSupv else 0
    stim_func2 = lambda t: cos[int(t/dt)] if t<tSupv else 0
    return stim_func1, stim_func2    

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=100, w=2*np.pi, tSupv=0.1,
    m=Uniform(20, 40), i=Uniform(-1, 1), stim_func1=lambda t: 0, stim_func2=lambda t: 0,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), f1=DoubleExp(1e-3, 1e-1), f2=None,
    dB=None, eB=None, wB=None, d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, d2=None, 
    learn0=False, learn1=False, learnB=False, test=False,
    eRate=1e-6, dRate=3e-6):

    tauRise, tauFall = (-1.0 / np.array(fTarget.poles))
    def feedback(state):
        x = state[0]
        y = state[1]
        state = state.reshape(-1,1)
        r = np.maximum(np.sqrt(x**2 + y**2), 1e-9)
        fr = (1-r**2)/(r**2)
        xdot = (r*fr*x + w*y)
        ydot = (r*fr*y - w*x)
        deriv = np.array([xdot, ydot]).reshape(-1,1)
        j11 = (-2*x**4 + y**2 - 3*x**2*y**2 - y**4)/(r**3)
        j12 = w - (x*y*(1 + x**2 + y**2))/(r**3)
        j21 = -w - (x*y*(1 + x**2 + y**2))/(r**3)
        j22 = (-x**4 - 2*y**4 + x**2*(1 - 3*y**2))/(r**3)
        second_deriv = (np.array([[j11, j12],[j21, j22]]) @ deriv).reshape(-1,1)
        feedback = state + (tauRise+tauFall)*deriv + (tauRise*tauFall)*second_deriv
        return feedback.ravel()

    weightsBias = wB if (np.any(wB) and not learnB) else np.zeros((nPre, nEns))
    weightsFF = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weightsFB = w1 if (np.any(w1) and not learn0 and not learn1) else np.zeros((nEns, nEns))
    with nengo.Network(seed=seed) as model:
        inpt1 = nengo.Node(stim_func1)
        inpt2 = nengo.Node(stim_func2)
        const = nengo.Node(1)
        inpt = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        bias = nengo.Ensemble(nPre, 2, max_rates=m, neuron_type=ReLu(), seed=seed)
        pre = nengo.Ensemble(nPre, 2, max_rates=m, neuron_type=ReLu(), seed=seed)
        tarA = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)
        tarA2 = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)
        tarX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarX2 = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarX3 = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarAX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarA2X = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        preX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        ens = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt1, inpt[0], synapse=None)
        nengo.Connection(inpt2, inpt[1], synapse=None)
        connStim = nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarX, synapse=fTarget)
        nengo.Connection(tarX, tarX2, synapse=fTarget, function=feedback)
        nengo.Connection(tarX2, tarX3, synapse=fTarget, function=feedback)
        nengo1d = nengo.Connection(pre, preX, synapse=fTarget, seed=seed)
        nengo1w = nengo.Connection(pre, tarA, synapse=fTarget, seed=seed)
        nengo2w = nengo.Connection(tarA, tarA2, synapse=fTarget, function=feedback, seed=seed)
        nengo2d = nengo.Connection(tarA, tarAX, synapse=fTarget, function=feedback, seed=seed)
        nengo2d = nengo.Connection(tarA2, tarA2X, synapse=fTarget, function=feedback, seed=seed)
        connFF = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weightsFF, weights=True), seed=seed)
        nengo.Connection(bias, tarA, synapse=fTarget, seed=seed)
        nengo.Connection(bias, tarA2, synapse=fTarget, seed=seed)
        connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
        connBias2 = nengo.Connection(bias, ens2, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)

        if learnB:  # learn a bias connection that feeds spikes representing zero to all ensembles
            nengo1w.transform = 0
            nodeBias = LearningNode(bias, ens, 1, conn=connBias, d=dB, e=eB, w=wB, eRate=eRate, dRate=dRate)
            nengo.Connection(bias.neurons, nodeBias[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeBias[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeBias[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(const, nodeBias[-1:], synapse=fTarget)
            nengo.Connection(nodeBias, ens.neurons, synapse=None)            

        if learn0:  # learn to receive supervised "recurrent" input from ReLU
            nodeFF = LearningNode(pre, ens, 2, conn=connFF, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, nodeFF[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeFF[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeFF[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(tarX, nodeFF[-2:], synapse=None)
            nengo.Connection(nodeFF, ens.neurons, synapse=None)

        if learn1: # learn to receive supervised "recurrent" input from neuron_type
            connSupv = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            nodeSupv = LearningNode(ens, ens2, 2, conn=connSupv, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, nodeSupv[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, nodeSupv[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA2.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(nodeSupv, ens2.neurons, synapse=None)

        if test:
            connFB = nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(weightsFB, weights=True))  # recurrent
            nengo3w = nengo.Connection(tarA, tarA, synapse=fTarget, function=feedback, seed=seed)
            

        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pPreX = nengo.Probe(preX, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarA2 = nengo.Probe(tarA2.neurons, synapse=None)
        pTarX = nengo.Probe(tarX, synapse=None)
        pTarX2 = nengo.Probe(tarX2, synapse=None)
        pTarX3 = nengo.Probe(tarX3, synapse=None)
        pTarAX = nengo.Probe(tarAX, synapse=None)
        pTarA2X = nengo.Probe(tarA2X, synapse=None)

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
        preX=sim.data[pPreX],
        ens=sim.data[pEns],
        ens2=sim.data[pEns2],
        tarA=sim.data[pTarA],
        tarA2=sim.data[pTarA2],
        tarX=sim.data[pTarX],
        tarX2=sim.data[pTarX2],
        tarX3=sim.data[pTarX3],
        tarAX=sim.data[pTarAX],
        tarA2X=sim.data[pTarA2X],
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

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, tTrans=5, seed=0, w=2*np.pi,
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), penalty=0, reg=1e-2, nBins=20,
    load=[]):

    print(f'Neuron type: {neuron_type}')
    rng = np.random.RandomState(seed=seed)

    if 0 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        dB, eB, wB = data['dB'], data['eB'], data['wB']
    else:
        print('train dB, eB, wB from bias to ens')
        dB, eB, wB = None, None, None
        for n in range(nTrain):
            data = go(neuron_type, learnB=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, w=w,
                dB=dB, eB=eB, wB=wB,
                fTarget=fTarget, fSmooth=fSmooth)
            dB, eB, wB = data['dB'], data['eB'], data['wB']
            np.savez(f"data/oscillate_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillate", neuron_type, "bias", n, nTrain)

    if 1 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens (kick input)')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, phase=n/nTrain, w=w)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, w=w,
                wB=wB,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/oscillate_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillate", neuron_type, "ens", n, nTrain)

    if 2 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('train d1 and f1 for ens to compute the feedback function oscillator')
        targets = np.zeros((nTrain, int(tTrain/dt), 2))
        targets2 = np.zeros((nTrain, int(tTrain/dt), 2))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, phase=n/nTrain, w=w)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, w=w,
                wB=wB,
                w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            targets[n] = data['tarX2']
            targets2[n] = fTarget.filt(data['tarX'],dt=dt)
            spikes[n] = data['ens']
        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrain,
            dt=dt, network="oscillate", neuron_type=neuron_type, ens=f"ens",
            penalty=penalty, seed=seed, reg=reg)
        f1 = DoubleExp(tauRise1, tauFall1)
        np.savez(f"data/oscillate_{neuron_type}.npz",
            dB=dB, eB=eB, wB=wB,
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)
        times = data['times']
        tarX = data['tarX']
        tarX2 = data['tarX2']
        tarAX = data['tarAX']
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        fig, ax = plt.subplots(figsize=((12, 4)))
        ax.plot(data['times'], tarX, color='gray', linewidth=0.5, label='input')
        ax.plot(data['times'], tarX2, color='k', linewidth=0.5, label='target')
        # ax.plot(data['times'], tarAX, linewidth=0.5, label='nengo')
        ax.plot(data['times'], xhat, linewidth=0.5, label='xhat')
        ax.axhline(1, linewidth=0.5)
        ax.axhline(-1, linewidth=0.5)
        ax.legend()
        ax.set(xlim=((0, tTrain)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/oscillate/decode_{neuron_type}.pdf')
    print(f"taus: {tauRise1:.4f}, {tauFall1:.4f}")

    if 3 in load:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from ens to ens2')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, phase=n/nTrain, w=w)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, w=w,
                wB=wB,
                w0=w0,
                d1=d1, f1=f1,
                e1=e1, w1=w1,
                fTarget=fTarget, fSmooth=fSmooth)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/oscillate_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
                "oscillate", neuron_type, "ens2", n, nTrain)

        print('check ens to ens2 connection')
        stim_func1, stim_func2 = makeSignal(10, phase=0.5, w=w)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn1=True, eRate=0,
            nEns=nEns, t=10, dt=dt, seed=seed, w=w,
            wB=wB,
            w0=w0,
            d1=d1, f1=f1,
            e1=e1, w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
            "oscillate", neuron_type, "ens2", -1, 0)
        times = data['times']
        tarX = data['tarX']
        tarX2 = data['tarX2']
        tarX3 = data['tarX3']
        aEns = f1.filt(data['ens'], dt=dt)
        aEns2 = f1.filt(data['ens2'], dt=dt)
        xhat = np.dot(aEns, d1)
        xhat2 = np.dot(aEns2, d1)
        tarAX = data['tarAX']
        tarA2X = data['tarA2X']
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=((12, 6)), sharex=True)
        ax.plot(data['times'], tarX2, linewidth=0.5, label='target')
        ax.plot(data['times'], xhat, linewidth=0.5, label='xhat')
        ax2.plot(data['times'], tarX3, linewidth=0.5, label='target')
        ax2.plot(data['times'], xhat2, linewidth=0.5, label='xhat')
        ax.axhline(1, linewidth=0.5)
        ax.axhline(-1, linewidth=0.5)
        ax2.axhline(1, linewidth=0.5)
        ax2.axhline(-1, linewidth=0.5)
        ax.set(xlim=((0, 10)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}_1(t))$")
        ax2.set(xlim=((0, 10)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}_2(t))$")
        fig.savefig(f'plots/oscillate/feedforward_{neuron_type}.pdf')

    dfs = []
    columns = ('neuron_type', 'n', 'error rmse', 'error freq', 'dimension')
    print('estimating error')
    for n in range(nTest):
        stim_func1, stim_func2 = makeKick(tTest+tTrans, phase=0, tSupv=0.2*n/nTest, w=w)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, test=True,
            nEns=nEns, t=tTest+tTrans, dt=dt, seed=seed, w=w,
            wB=wB,
            w0=w0,
            f1=f1,
            w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        times = data['times']
        timesError = times[int(tTrans/dt):]
        preX = data['preX']
        # tarX = data['tarX']
        # tarX2 = data['tarX2']
        tarX = np.concatenate((np.sin(w*times).reshape(-1,1), np.cos(w*times).reshape(-1,1)), axis=1)
        tarAX = data['tarAX']
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        aTarA = fSmooth.filt(data['tarA'], dt=dt)
        minRate = np.min(np.max(aEns[int(tTrans/dt):], axis=0))
        maxRate = np.max(aEns[int(tTrans/dt):])
        print(f'{neuron_type} firing rates: {minRate:.0f} to {maxRate:.0f}Hz')

        errorRMSE0, errorFreq0, freq0, phase0, mag0, base0 = fitSinusoid(xhat[int(tTrans/dt):,0], neuron_type, dt=dt)
        errorRMSE1, errorFreq1, freq1, phase1, mag1, base1 = fitSinusoid(xhat[int(tTrans/dt):,1], neuron_type, dt=dt)
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE0, errorFreq0, '0']], columns=columns))
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE1, errorFreq1, '1']], columns=columns))
        # fit0 = mag0*np.sin(freq0*(timesError+phase0))+np.ones_like(timesError)*base0
        # fit1 = mag1*np.sin(freq1*(timesError+phase1))+np.ones_like(timesError)*base1

        fig, ax = plt.subplots(figsize=((5.25, 1.5)))
        # ax.plot(timesError, fit0, linewidth=0.5, label=f'target, w={freq0:.2f}')
        # ax.plot(timesError, fit1, linewidth=0.5, label=f'target, w={freq1:.2f}')
        ax.plot(times, xhat, linewidth=0.5, label='xhat')
        ax.legend(loc='upper right')
        ax.set(xlim=((0, tTest+tTrans)), yticks=((-1, 1)), xticks=(()), xlabel='', ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/oscillate/test_{neuron_type}_{n}.pdf')

    return times, tarX, xhat, dfs

def compare(neuron_types, nTrain=10, tTrain=10, nTest=10, tTest=10, tTrans=5, load=[],
    eRates=[1e-6, 1e-5, 1e-6, 1e-7]):

    dfsAll = []
    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=((5.25, 3)))
    for i, neuron_type in enumerate(neuron_types):
        times, tarX, xhat, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest,
            tTrans=tTrans, eRate=eRates[i], load=load)
        dfsAll.extend(dfs)
        ax.plot(times, xhat[:,0], label=f"{str(neuron_type)[:-2]}", linewidth=0.5)
        ax2.plot(times, xhat[:,1], label=f"{str(neuron_type)[:-2]}", linewidth=0.5)
    df = pd.concat([df for df in dfsAll], ignore_index=True)

    ax.plot(times, tarX[:,0], label='target', color='k', linewidth=0.5)
    ax2.plot(times, tarX[:,1], label='target', color='k', linewidth=0.5)
    ax.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}_1(t))$")
    ax2.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}_2(t))$")
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate.pdf')
    fig.savefig('plots/figures/oscillate.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='neuron_type', y='error rmse', ax=ax)
    ax.set(xlabel='', ylim=((0, 0.1)), yticks=((0, 0.1)), ylabel='Error (RMSE)')
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_barplot_rmse.pdf')
    fig.savefig('plots/figures/oscillate_barplot_rmse.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='neuron_type', y='error freq', ax=ax)
    ax.set(xlabel='', ylim=((0, 0.15)), yticks=((0, 0.15)), ylabel=r'Error ($\omega$)')
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate_barplot_freq.pdf')
    fig.savefig('plots/figures/oscillate_barplot_freq.svg')

def print_time_constants():
    for neuron_type in ['LIF()', 'Izhikevich()', 'Wilson()', 'Pyramidal()']:
        data = np.load(f"data/oscillate_{neuron_type}.npz")
        rise, fall = 1000*data['tauRise1'], 1000*data['tauFall1']
        print(f"{neuron_type}:  \t rise {rise:.3}, fall {fall:.4}")
print_time_constants()

# compare([LIF(), Izhikevich(), Wilson(), NEURON("Pyramidal")], load=[0,1,2,3])