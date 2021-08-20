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

def makeSignal(t, fTarget, seed, nTrain=10, mag=1.2, w=2*np.pi, high=1, rms=1e-3, dt=1e-3):
    # Create an oscillator with the correct frequency, 
    tauRise = (-1.0 / np.array(fTarget.poles))[0]
    tauFall = (-1.0 / np.array(fTarget.poles))[1]
    # account for finite dt and tau when calculating the A matrix necessary on the recurrent transform
    # see https://forum.nengo.ai/t/oscillator-example/513/2)
    fTarget = (cont2discrete(Lowpass(tauRise), dt=dt) * cont2discrete(Lowpass(tauFall), dt=dt))
    idealA= [[0, w], [-w, 0]]
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
        sim.run(t+2+dt, progress_bar=False)
    phase = seed * w/nTrain  # 0-2pi
    phase_idx = int((phase / w) / dt)  # 0-1s in steps of dt
    tar1 = mag * sim.data[pTarX][phase_idx:,0]
    tar2 = mag * sim.data[pTarX][phase_idx:,1]
    scale = 1 + rms*np.cos(high*sim.trange()[phase_idx:])
    # fig, ax = plt.subplots()
    # ax.plot(tar1*scale)
    # ax.plot(tar2*scale)
    # fig.savefig(f'plots/oscillate7/signal_{seed}.pdf')
    # raise
    stim_func1 = lambda t: tar1[int(t/dt)] * scale[int(t/dt)]
    stim_func2 = lambda t: tar2[int(t/dt)] * scale[int(t/dt)]
    return stim_func1, stim_func2


def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=100, w=2*np.pi,
    m=Uniform(30, 60), i=Uniform(-0.3, 0.3), stim_func1=lambda t: 0, stim_func2=lambda t: 0, tSupv=0.1,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, learn0=False, learn1=False, test=False,
    eRate=1e-6, dRate=3e-6):

    tauRise = (-1.0 / np.array(fTarget.poles))[0]
    tauFall = (-1.0 / np.array(fTarget.poles))[1]
    fTarget = (cont2discrete(Lowpass(tauRise), dt=dt) * cont2discrete(Lowpass(tauFall), dt=dt))
    idealA= [[0, w], [-w, 0]]
    dsys = cont2discrete(LinearSystem((idealA, [[1], [0]], [[1, 0]], [[0]])), dt=dt)
    simA = ss2sim(dsys, fTarget, dt=None).ss[0]

    weightsFF = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weightsFB = w1 if (np.any(w1) and not learn0 and not learn1) else np.zeros((nEns, nEns))
    with nengo.Network() as model:
        inpt1 = nengo.Node(stim_func1)
        inpt2 = nengo.Node(stim_func2)
        inpt = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarA = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        tarA2 = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        tarX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarX2 = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarAX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarA2X = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        pre = nengo.Ensemble(nPre, 2, radius=1.5, neuron_type=ReLu(), seed=seed)
        pre2 = nengo.Ensemble(nPre, 2, radius=1.5, neuron_type=ReLu(), seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt1, inpt[0], synapse=None)
        nengo.Connection(inpt2, inpt[1], synapse=None)
        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(pre, tarA, synapse=fTarget)
        nengo.Connection(inpt, tarX, synapse=fTarget)
        nengo.Connection(tarA, tarAX, synapse=fTarget, transform=simA)
        nengo.Connection(tarA, tarA2, synapse=fTarget, transform=simA)
        nengo.Connection(tarA2, tarA2X, synapse=fTarget)
        # nengo.Connection(tarX, pre2, synapse=fTarget, transform=simA)
        # nengo.Connection(tarX, tarA2, synapse=fTarget, transform=simA)
        nengo.Connection(tarX, tarX2, synapse=fTarget, transform=simA)
        connFF = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weightsFF, weights=True))

        if learn0:  # learn to receive supervised "recurrent" input from ReLU
            nodeFF = LearningNode(pre, ens, 2, conn=connFF, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, nodeFF[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeFF[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeFF[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt1, nodeFF[-2], synapse=fTarget)
            nengo.Connection(inpt2, nodeFF[-1], synapse=fTarget)
            nengo.Connection(nodeFF, ens.neurons, synapse=None)

        if learn1: # learn to receive supervised "recurrent" input from neuron_type
            # ens3 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
            # connFF2 = nengo.Connection(pre2, ens3, synapse=fTarget, solver=NoSolver(weightsFF, weights=True))
            # pEns3 = nengo.Probe(ens3.neurons, synapse=None)
            connSupv = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            nodeSupv = LearningNode(ens, ens2, 2, conn=connSupv, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, nodeSupv[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, nodeSupv[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(ens.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            # nengo.Connection(ens3.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            # nengo.Connection(tarA2.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(nodeSupv, ens2.neurons, synapse=None)

        if test:
            connFB = nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(weightsFB, weights=True))  # recurrent
            nengo.Connection(tarA, tarA, synapse=fTarget, transform=simA)
            off = nengo.Node(lambda t: 0 if t<=tSupv else -1e4)
            nengo.Connection(off, pre.neurons, synapse=None, transform=np.ones((nPre, 1)))


        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pPre2 = nengo.Probe(pre2.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarA2 = nengo.Probe(tarA2.neurons, synapse=None)
        pTarAX = nengo.Probe(tarAX, synapse=None)
        pTarA2X = nengo.Probe(tarA2X, synapse=None)
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
        pre2=sim.data[pPre2],
        ens=sim.data[pEns],
        ens2=sim.data[pEns2],
        # ens3=sim.data[pEns3] if learn1 else None,
        tarA=sim.data[pTarA],
        tarA2=sim.data[pTarA2],
        tarAX=sim.data[pTarAX],
        tarA2X=sim.data[pTarA2X],
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

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, tTransTrain=0, tTrans=0, nTrainDF=10,
    nEns=100, dt=1e-3, tSupv=0.15, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1),
    load=[]):

    print(f'Neuron type: {neuron_type}')

    if 0 in load:
        data = np.load(f"data/oscillate7_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens (kick input)')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, fTarget, n, nTrain=nTrain)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/oscillate7_{neuron_type}.npz", d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillate7", neuron_type, "ens", n, nTrain)

        print('check pre to ens connection')
        stim_func1, stim_func2 = makeSignal(tTrain, fTarget, 0)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn0=True, eRate=0, dRate=0,
        # data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2,
            nEns=nEns, t=tTrain, dt=dt,
            d0=d0, e0=e0, w0=w0,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "oscillate7", neuron_type, "ens", -1, 0)

        # fig, ax = plt.subplots()
        # # ax.plot(data['times'], data['tarX'], label='X')
        # ax.plot(data['times'], fTarget.filt(data['tarX'], dt=dt), label='X')
        # ax.plot(data['times'], data['tarAX'], label='tarAX')
        # # ax.plot(data['times'], data['tarX2'], label='X2')
        # ax.legend()
        # fig.savefig('plots/oscillate7/tar.pdf')
        # raise

    if 1 in load:
        data = np.load(f"data/oscillate7_{neuron_type}.npz")
        d1, tauRise, tauFall = data['d1'], data['tauRise'], data['tauFall']
        f1 = DoubleExp(tauRise, tauFall)
    else:
        print('train d1 and f1 for ens to compute the A matrix for the oscillator')
        targets = np.zeros((nTrainDF, int(tTrain/dt), 2))
        spikes = np.zeros((nTrainDF, int(tTrain/dt), nEns))
        for n in range(nTrainDF):
            stim_func1, stim_func2 = makeSignal(tTrain, fTarget, n, nTrain=nTrain)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain+tTransTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            # targets[n] = fTarget.filt(data['tarX'], dt=dt)[int(tTransTrain/dt):]
            # targets[n] = fTarget.filt(data['tarX2'], dt=dt)[int(tTransTrain/dt):]
            # targets[n] = data['tarX2'][int(tTransTrain/dt):]
            # targets[n] = data['tarX'][int(tTransTrain/dt):]
            targets[n] = data['tarAX'][int(tTransTrain/dt):]
            # targets[n] = data['pre2X'][int(tTransTrain/dt):]
            # targets[n] = data['tarA2X'][int(tTransTrain/dt):]
            spikes[n] = data['ens'][int(tTransTrain/dt):]
        d1, tauRise, tauFall = trainDF(spikes, targets, nTrainDF,
            dt=dt, network="oscillate7", neuron_type=neuron_type, ens="ens")
        f1 = DoubleExp(tauRise, tauFall)
        np.savez(f"data/oscillate7_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise=tauRise, tauFall=tauFall)

        times = data['times']
        # tarX = fTarget.filt(data['tarX'], dt=dt)
        # tarX = data['tarX2']
        tarX = data['tarAX']
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        fig, ax = plt.subplots(figsize=((5.25, 1.5)))
        ax.plot(data['times'], tarX, color='k', linewidth=0.5)
        ax.plot(data['times'], xhat, linewidth=0.5)
        ax.set(xlim=((0, tTrain)), xticks=(()), ylim=((-1.2, 1.2)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/oscillate7/decode_{neuron_type}.pdf')

    print(f"taus: {tauRise:.4f}, {tauFall:.4f}")

    if 2 in load:
        data = np.load(f"data/oscillate7_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from ens to ens2')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, fTarget, n, nTrain=nTrain)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn1=True, eRate=3*eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, f1=f1,
                e1=e1, w1=w1,
                fTarget=fTarget, fSmooth=fSmooth)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/oscillate7_{neuron_type}.npz",
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise=tauRise, tauFall=tauFall,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
            # plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
                "oscillate7", neuron_type, "ens2", n, nTrain)

        print('check ens to ens2 connection')
        stim_func1, stim_func2 = makeSignal(tTrain, fTarget, n)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn1=True, eRate=0,
            nEns=nEns, t=tTrain, dt=dt,
            d0=d0, e0=e0, w0=w0,
            d1=d1, f1=f1,
            e1=e1, w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        # plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
        plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
            "oscillate7", neuron_type, "ens2", -1, 0)
        times = data['times']
        tarX = data['tarAX']
        # tarX2 = data['tarA2X']
        aEns = f1.filt(data['ens'], dt=dt)
        aEns2 = f1.filt(data['ens2'], dt=dt)
        xhat = np.dot(aEns, d1)
        xhat2 = np.dot(aEns2, d1)
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=((5.25, 3)), sharex=True)
        ax.plot(data['times'], tarX, linewidth=0.5)
        ax.plot(data['times'], xhat, linewidth=0.5)
        ax2.plot(data['times'], tarX, linewidth=0.5)
        ax2.plot(data['times'], xhat2, linewidth=0.5)
        ax.set(xlim=((0, tTrain)), xticks=(()), ylim=((-1, 1)), ylabel=r"X")
        ax2.set(xlim=((0, tTrain)), xticks=(()), ylim=((-1, 1)), ylabel=r"X2")
        fig.savefig(f'plots/oscillate7/check_{neuron_type}.pdf')

    dfs = []
    columns = ('neuron_type', 'n', 'error rmse', 'error freq', 'dimension')
    print('estimating error')
    for n in range(nTest):
        stim_func1, stim_func2 = makeSignal(tTest+tTrans, fTarget, n, nTrain=nTest, rms=1e-3, mag=1.0)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, test=True,
            nEns=nEns, t=tTest+tTrans, dt=dt, tSupv=tSupv,
            w0=w0,
            f1=f1,
            w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        times = data['times']
        # tarX = fTarget.filt(data['inpt'], dt=dt)
        tarX = data['tarX']
        aEns = f1.filt(data['ens'], dt=dt)
        aTarA = fSmooth.filt(data['tarA'], dt=dt)
        xhat = np.dot(aEns, d1)
        tarAX = data['tarAX']
        print(f'observed firing rate range: {np.min(np.max(aEns[1000:], axis=0)):.0f} to {np.max(aEns[1000:]):.0f}Hz')
        print(f'tarA firing rate range: {np.min(np.max(aTarA[1000:], axis=0)):.0f} to {np.max(aTarA[1000:]):.0f}Hz')

        fig, ax = plt.subplots(figsize=((5.25, 1.5)))
        ax.plot(data['times'], tarX, color='k', linewidth=0.5, label='tar')
        ax.plot(data['times'], tarAX, linewidth=0.5, label='tarA')
        ax.plot(data['times'], xhat, linewidth=0.5, label='xhat')
        ax.legend(loc='upper right')
        ax.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/oscillate7/test_{neuron_type}.pdf')

        errorRMSE0, errorFreq0, freq0, phase0, mag0, base0 = fitSinusoid(xhat[int(tTrans/dt):,0], neuron_type, dt=dt)  # muFreq=muFreq, sigmaFreq=sigmaFreq, base=base
        errorRMSE1, errorFreq1, freq1, phase1, mag1, base1 = fitSinusoid(xhat[int(tTrans/dt):,1], neuron_type, dt=dt)  # muFreq=muFreq, sigmaFreq=sigmaFreq, base=base
        # error0 = rmse(xhat[int(tTrans/dt):,0], tarX[int(tTrans/dt):,0])
        # error1 = rmse(xhat[int(tTrans/dt):,1], tarX[int(tTrans/dt):,1])
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE0, errorFreq0, '0']], columns=columns))
        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE1, errorFreq1, '1']], columns=columns))

    return times, tarX, xhat, [freq0, freq1], dfs

def compare(neuron_types, nTrain=10, nTrainDF=10, tTrain=20, nTest=1, tTest=10, tTrans=10, load=[],
    eRates=[1e-7, 1e-7, 1e-7, 3e-8], tTransTrains=[0, 5, 5, 10]):

    dfsAll = []
    fig, ax = plt.subplots(figsize=((5.25, 1.5)))
    fig2, ax2 = plt.subplots(figsize=((5.25, 1.5)))
    for i, neuron_type in enumerate(neuron_types):
        times, tarX, xhat, freq, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest,
            nTrainDF=nTrainDF, eRate=eRates[i], tTransTrain=tTransTrains[i], tTrans=tTrans, load=load)
        dfsAll.extend(dfs)
        ax.plot(times, xhat[:,0], label=f"{str(neuron_type)[:-2]}, " + r"$\omega=$" + f"{freq[0]:.2f}", linewidth=0.5)
        ax2.plot(times, xhat[:,1], label=f"{str(neuron_type)[:-2]}, " + r"$\omega=$" + f"{freq[1]:.2f}", linewidth=0.5)
    df = pd.concat([df for df in dfsAll], ignore_index=True)

    ax.plot(times, tarX[:,0], label='target, ' + r"$\omega=$" + f"{2*np.pi:.2f}", color='k', linewidth=0.5)
    # ax.axvline(tKick, color='k', linestyle=":", label="tKick", linewidth=0.5)
    ax.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig.savefig('plots/figures/oscillate7_dim1.pdf')
    fig.savefig('plots/figures/oscillate7_dim1.svg')

    ax2.plot(times, tarX[:,1], label='target, ' + r"$\omega=$" + f"{2*np.pi:.2f}", color='k', linewidth=0.5)
    # ax2.axvline(tKick, color='k', linestyle=":", label="tKick", linewidth=0.5)
    ax2.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax2.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig2.savefig('plots/figures/oscillate7_dim2.pdf')
    fig2.savefig('plots/figures/oscillate7_dim2.svg')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    # sns.barplot(data=df, x='neuron_type', y='error rmse', ax=ax)
    # ax.set(xlabel='', ylim=((0, 0.2)), yticks=((0, 0.2)), ylabel='Error (RMSE)')
    # plt.tight_layout()
    # fig.savefig('plots/figures/oscillate_barplot_rmse.pdf')
    # fig.savefig('plots/figures/oscillate_barplot_rmse.svg')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    # sns.barplot(data=df, x='neuron_type', y='error freq', ax=ax)
    # ax.set(xlabel='', ylim=((0, 0.1)), yticks=((0, 0.1)), ylabel=r'Error ($\omega$)')
    # plt.tight_layout()
    # fig.savefig('plots/figures/oscillate_barplot_freq.pdf')
    # fig.savefig('plots/figures/oscillate_barplot_freq.svg')


compare([LIF()], nTrain=10, nTrainDF=10, eRates=[1e-7], load=[0,1,2])