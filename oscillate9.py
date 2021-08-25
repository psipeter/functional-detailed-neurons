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

def makeSignal(t, phase, w=2*np.pi, mag=0.9, dt=1e-3):
    idx = int(phase * t / dt)
    times = np.arange(0, 3*t, dt)[idx:]
    envelope = 1 + mag * np.sin(w/t*times)
    sin = np.sin(w*times)
    cos = np.cos(w*times)
    stim_func1 = lambda t: envelope[int(t/dt)] * sin[int(t/dt)]
    stim_func2 = lambda t: envelope[int(t/dt)] * cos[int(t/dt)]
    return stim_func1, stim_func2

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=100, w=2*np.pi,
    rate=0, i=Uniform(-0.5, 0.5), stim_func1=lambda t: 0, stim_func2=lambda t: 0, tSupv=0.1,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, learn0=False, learn1=False, test=False,
    eRate=1e-6, dRate=3e-6):

    tauRise, tauFall = (-1.0 / np.array(fTarget.poles))
    m = Uniform(rate/2, rate)
    w = 2*np.pi
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

    weightsFF = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weightsFB = w1 if (np.any(w1) and not learn0 and not learn1) else np.zeros((nEns, nEns))
    with nengo.Network(seed=seed) as model:
        inpt1 = nengo.Node(stim_func1)
        inpt2 = nengo.Node(stim_func2)
        inpt = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarA = nengo.Ensemble(nEns, 2, radius=0.8, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)
        tarA2 = nengo.Ensemble(nEns, 2, radius=0.8, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)
        tarX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarX2 = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarFB = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tarAX = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        pre = nengo.Ensemble(nPre, 2, radius=2, neuron_type=ReLu(), seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt1, inpt[0], synapse=None)
        nengo.Connection(inpt2, inpt[1], synapse=None)
        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarX, synapse=fTarget, seed=seed)
        nengo.Connection(pre, tarA, synapse=fTarget, seed=seed)
        nengo.Connection(tarX, tarX2, synapse=fTarget, function=feedback)
        nengo.Connection(tarX2, tarA2, synapse=None, function=None)
        nengo.Connection(tarA, tarAX, synapse=fTarget, seed=seed)
        connFF = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weightsFF, weights=True), seed=seed)

        if learn0:  # learn to receive supervised "recurrent" input from ReLU
            nodeFF = LearningNode(pre, ens, 2, conn=connFF, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, nodeFF[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeFF[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeFF[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(tarX, nodeFF[-2:], synapse=None)
            nengo.Connection(nodeFF, ens.neurons, synapse=None)

        if learn1: # learn to receive supervised "recurrent" input from neuron_type
            # pre3 = nengo.Ensemble(nPre, 2, radius=2, neuron_type=ReLu(), seed=seed)
            # ens3 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
            # nengo.Connection(tarX, pre3, synapse=None, function=feedback)
            # connFF2 = nengo.Connection(pre3, ens3, synapse=fTarget, solver=NoSolver(weightsFF, weights=True), seed=seed)
            # pEns3 = nengo.Probe(ens3.neurons, synapse=None)
            connSupv = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            nodeSupv = LearningNode(ens, ens2, 2, conn=connSupv, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, nodeSupv[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, nodeSupv[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA2.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            # nengo.Connection(ens3.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(nodeSupv, ens2.neurons, synapse=None)

        if test:
            connFB = nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(weightsFB, weights=True))  # recurrent
            nengo.Connection(tarA, tarA, synapse=fTarget, function=feedback, seed=seed)
            nengo.Connection(tarFB, tarFB, synapse=fTarget, function=feedback)
            off = nengo.Node(lambda t: 0 if t<=tSupv else -1e4)
            nengo.Connection(off, pre.neurons, synapse=None, transform=np.ones((nPre, 1)))
            nengo.Connection(pre, tarFB, synapse=fTarget)
            # kick = nengo.Node(lambda t: [1, -1] if t<tSupv else [0,0])
            # nengo.Connection(kick, tarFB, synapse=None)
            # nengo.Connection(kick, tarA, synapse=None)


        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarA2 = nengo.Probe(tarA2.neurons, synapse=None)
        pTarX = nengo.Probe(tarX, synapse=None)
        pTarX2 = nengo.Probe(tarX2, synapse=None)
        pTarAX = nengo.Probe(tarAX, synapse=None)
        pTarFB = nengo.Probe(tarFB, synapse=fTarget)

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
        # ens3=sim.data[pEns3] if learn1 else None,
        tarA=sim.data[pTarA],
        tarA2=sim.data[pTarA2],
        tarX=sim.data[pTarX],
        tarX2=sim.data[pTarX2],
        tarAX=sim.data[pTarAX],
        tarFB=sim.data[pTarFB],
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
        f1=f1,
    )

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, tTransTrain=0, tTrans=0, nTrainDF=10, seed=0, rate=60,
    nEns=100, dt=1e-3, tSupv=0.1, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), penalty=0,
    load=[]):

    print(f'Neuron type: {neuron_type}')

    if 0 in load:
        data = np.load(f"data/oscillate9_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens (kick input)')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, n/nTrain)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, rate=rate,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/oscillate9_{neuron_type}_rate{rate}.npz", d0=d0, e0=e0, w0=w0)
            # plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            #     "oscillate9", neuron_type, "ens", n, nTrain)

        # print('check pre to ens connection')
        # stim_func1, stim_func2 = makeSignal(tTrain, 0)
        # data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn0=True, eRate=0, dRate=0,
        #     nEns=nEns, t=tTrain, dt=dt, seed=seed, rate=rate,
        #     d0=d0, e0=e0, w0=w0,
        #     fTarget=fTarget, fSmooth=fSmooth)
        # plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
        #     "oscillate9", neuron_type, "ens", -1, 0)

    if 1 in load:
        data = np.load(f"data/oscillate9_{neuron_type}.npz")
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        d2, tauRise2, tauFall2 = data['d2'], data['tauRise2'], data['tauFall2']
        f1 = DoubleExp(tauRise1, tauFall1)
        f2 = DoubleExp(tauRise2, tauFall2)
    else:
        print('train d1 and f1 for ens to compute the feedback function oscillator')
        print('train d2 and f2 for ens to readout the representation')
        targets = np.zeros((nTrainDF, int(tTrain/dt), 2))  # 
        targets2 = np.zeros((nTrainDF, int(tTrain/dt), 2))
        spikes = np.zeros((nTrainDF, int(tTrain/dt), nEns))
        for n in range(nTrainDF):
            stim_func1, stim_func2 = makeSignal(tTrain, n/nTrain)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain+tTransTrain, dt=dt, seed=seed, rate=rate,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            targets[n] = data['tarX2'][int(tTransTrain/dt):]
            targets2[n] = fTarget.filt(data['tarX'],dt=dt)[int(tTransTrain/dt):]
            spikes[n] = data['ens'][int(tTransTrain/dt):]
        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrainDF,
            dt=dt, network="oscillate9", neuron_type=neuron_type, ens=f"ens", penalty=penalty, seed=seed, reg=1e-2)
        d2, tauRise2, tauFall2 = trainDF(spikes, targets2, nTrainDF,
            dt=dt, network="oscillate9", neuron_type=neuron_type, ens=f"ens", penalty=penalty, seed=seed,)
        f1 = DoubleExp(tauRise1, tauFall1)
        f2 = DoubleExp(tauRise2, tauFall2)
        np.savez(f"data/oscillate9_{neuron_type}_rate{rate}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
            d2=d2, tauRise2=tauRise2, tauFall2=tauFall2)

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
        ax.set(xlim=((0, tTrain)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/oscillate9/decode_{neuron_type}_rate{rate}.pdf')

    print(f"taus: {tauRise1:.4f}, {tauFall1:.4f}")
    print(f"readout: {tauRise2:.4f}, {tauFall2:.4f}")

    if 2 in load:
        data = np.load(f"data/oscillate9_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from ens to ens2')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, n/nTrain)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn1=True, eRate=3*eRate,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, rate=rate,
                d0=d0, e0=e0, w0=w0,
                d1=d1, f1=f1,
                e1=e1, w1=w1,
                fTarget=fTarget, fSmooth=fSmooth)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/oscillate9_{neuron_type}_rate{rate}.npz",
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                d2=d2, tauRise2=tauRise2, tauFall2=tauFall2,
                e1=e1, w1=w1)
            # plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
                # "oscillate9", neuron_type, "ens2", n, nTrain)

        print('check ens to ens2 connection')
        stim_func1, stim_func2 = makeSignal(tTrain, 0)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn1=True, eRate=0,
            nEns=nEns, t=tTrain, dt=dt, seed=seed, rate=rate,
            d0=d0, e0=e0, w0=w0,
            d1=d1, f1=f1,
            e1=e1, w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        # plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
        # plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
            # "oscillate9", neuron_type, "ens2", -1, 0)
        times = data['times']
        tarX = data['tarX']
        tarX2 = data['tarX2']
        tarX3 = fTarget.filt(data['tarX2'], dt=dt)
        aEns = f1.filt(data['ens'], dt=dt)
        aEns2 = f2.filt(data['ens2'], dt=dt)
        xhat = np.dot(aEns, d1)
        xhat2 = np.dot(aEns2, d2)
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=((12, 6)), sharex=True)
        ax.plot(data['times'], tarX, linewidth=0.5)
        ax.plot(data['times'], tarX2, linewidth=0.5)
        ax.plot(data['times'], xhat, linewidth=0.5)
        ax2.plot(data['times'], tarX3, linewidth=0.5)
        ax2.plot(data['times'], xhat2, linewidth=0.5)
        ax.axhline(1, linewidth=0.5)
        ax.axhline(-1, linewidth=0.5)
        ax2.axhline(1, linewidth=0.5)
        ax2.axhline(-1, linewidth=0.5)
        ax.set(xlim=((0, tTrain)), xticks=(()), yticks=((-1, 1)), ylabel=r"X")
        ax2.set(xlim=((0, tTrain)), xticks=(()), yticks=((-1, 1)), ylabel=r"X2")
        fig.savefig(f'plots/oscillate9/feedforward_{neuron_type}_rate{rate}.pdf')

    dfs = []
    columns = ('neuron_type', 'n', 'error rmse', 'error freq', 'dimension')
    print('estimating error')
    for n in range(nTest):
        stim_func1, stim_func2 = makeSignal(tTest+tTrans, n/nTest, mag=0)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, test=True,
            nEns=nEns, t=tTest+tTrans, dt=dt, tSupv=tSupv, seed=seed, rate=rate,
            w0=w0,
            f1=f1,
            w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        times = data['times']
        tarX = data['tarX']
        tarX2 = data['tarX2']
        tarAX = data['tarAX']
        tarFB = data['tarFB']
        aEns = f2.filt(data['ens'], dt=dt)
        aTarA = fSmooth.filt(data['tarA'], dt=dt)
        xhat = np.dot(aEns, d2)
        print(f'observed firing rate range: {np.min(np.max(aEns[2000:], axis=0)):.0f} to {np.max(aEns[2000:]):.0f}Hz')
        print(f'tarA firing rate range: {np.min(np.max(aTarA[2000:], axis=0)):.0f} to {np.max(aTarA[2000:]):.0f}Hz')

        fig, ax = plt.subplots(figsize=((12, 4)))
        ax.plot(data['times'], tarX2, color='gray', linewidth=0.5, label='target (ff)')
        ax.plot(data['times'], tarFB, color='k', linewidth=0.5, label='target (fb)')
        ax.plot(data['times'], tarAX, linewidth=0.5, label='nengo')
        ax.plot(data['times'], xhat, linewidth=0.5, label='xhat')
        ax.axhline(1, linewidth=0.5)
        ax.axhline(-1, linewidth=0.5)
        ax.legend(loc='upper right')
        ax.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/oscillate9/test_{neuron_type}_rate{rate}.pdf')

        # errorRMSE0, errorFreq0, freq0, phase0, mag0, base0 = fitSinusoid(xhat[int(tTrans/dt):,0], neuron_type, dt=dt)
        # errorRMSE1, errorFreq1, freq1, phase1, mag1, base1 = fitSinusoid(xhat[int(tTrans/dt):,1], neuron_type, dt=dt)
        # dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE0, errorFreq0, '0']], columns=columns))
        # dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, errorRMSE1, errorFreq1, '1']], columns=columns))

    # return times, tarX, xhat, [freq0, freq1], dfs

def compare(neuron_types, nTrain=10, nTrainDF=10, tTrain=10, nTest=1, tTest=10, load=[],
    eRates=[1e-7, 3e-7, 1e-7, 3e-8]):

    dfsAll = []
    # fig, ax = plt.subplots(figsize=((5.25, 1.5)))
    # fig2, ax2 = plt.subplots(figsize=((5.25, 1.5)))
    for i, neuron_type in enumerate(neuron_types):
        run(neuron_type, nTrain, nTest, tTrain, tTest, nTrainDF=nTrainDF, eRate=eRates[i], load=load)
        return
    #     times, tarX, xhat, freq, dfs = run(neuron_type, nTrain, nTest, tTrain, tTest,
    #         nTrainDF=nTrainDF, eRate=eRates[i], load=load)
    #     dfsAll.extend(dfs)
    #     ax.plot(times, xhat[:,0], label=f"{str(neuron_type)[:-2]}, " + r"$\omega=$" + f"{freq[0]:.2f}", linewidth=0.5)
    #     ax2.plot(times, xhat[:,1], label=f"{str(neuron_type)[:-2]}, " + r"$\omega=$" + f"{freq[1]:.2f}", linewidth=0.5)
    # df = pd.concat([df for df in dfsAll], ignore_index=True)

    # ax.plot(times, tarX[:,0], label='target, ' + r"$\omega=$" + f"{2*np.pi:.2f}", color='k', linewidth=0.5)
    # ax.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    # ax.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # fig.savefig('plots/figures/oscillate9_dim1.pdf')
    # fig.savefig('plots/figures/oscillate9_dim1.svg')

    # ax2.plot(times, tarX[:,1], label='target, ' + r"$\omega=$" + f"{2*np.pi:.2f}", color='k', linewidth=0.5)
    # ax2.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    # ax2.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # fig2.savefig('plots/figures/oscillate9_dim2.pdf')
    # fig2.savefig('plots/figures/oscillate9_dim2.svg')

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


compare([LIF()], nTrain=5, nTrainDF=5, eRates=[3e-7], load=[])