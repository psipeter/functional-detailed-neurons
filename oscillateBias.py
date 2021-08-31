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

from neuron_types import LIF, Izhikevich, Wilson, Pyramidal, nrnReset
from utils import LearningNode, trainDF, fitSinusoid
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
palette = sns.color_palette('colorblind')
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

def makeSignal(t, phase, w=2*np.pi, aEnv=0.9, aSin=0.7, dt=1e-3):
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

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=100, w=2*np.pi, rate=0, tSupv=0.1,
    i=Uniform(-0.5, 0.5), stim_func1=lambda t: 0, stim_func2=lambda t: 0,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), f1=DoubleExp(1e-3, 1e-1), f2=None,
    dB=None, eB=None, wB=None, d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, d2=None, 
    learn0=False, check0=False, learn1=False, check1=False, learnB=False, checkB=False, test=False,
    eRate=1e-6, dRate=3e-6, regNengo=1e-2):

    tauRise, tauFall = (-1.0 / np.array(fTarget.poles))
    m = Uniform(rate/2, rate)
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
    weightsBias = wB if (np.any(wB) and not learnB) else np.zeros((nPre, nEns))
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
        nengo1w = nengo.Connection(pre, tarA, synapse=fTarget, seed=seed, solver=LstsqL2(weights=True, reg=regNengo))
        nengo2w = nengo.Connection(tarA, tarA2, synapse=fTarget, function=feedback, seed=seed, solver=LstsqL2(weights=True, reg=regNengo))
        nengo2d = nengo.Connection(tarA, tarAX, synapse=fTarget, function=feedback, seed=seed)
        nengo2d = nengo.Connection(tarA2, tarA2X, synapse=fTarget, function=feedback, seed=seed)
        connFF = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weightsFF, weights=True), seed=seed)
        nengo.Connection(bias, tarA, synapse=fTarget, seed=seed, solver=LstsqL2(weights=True, reg=regNengo))
        nengo.Connection(bias, tarA2, synapse=fTarget, seed=seed, solver=LstsqL2(weights=True, reg=regNengo))
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
        if checkB:
            assert np.any(weightsBias)  # connBias define above

        if learn0:  # learn to receive supervised "recurrent" input from ReLU
            nodeFF = LearningNode(pre, ens, 2, conn=connFF, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, nodeFF[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeFF[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeFF[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(tarX, nodeFF[-2:], synapse=None)
            nengo.Connection(nodeFF, ens.neurons, synapse=None)
        if check0:
            assert np.any(weightsFF)  # connFF define above

        if learn1: # learn to receive supervised "recurrent" input from neuron_type
            pre3 = nengo.Ensemble(nPre, 2, max_rates=m, neuron_type=ReLu(), seed=seed)
            ens3 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
            nengo.Connection(tarX, pre3, synapse=None, function=feedback)
            connFF2 = nengo.Connection(pre3, ens3, synapse=fTarget, solver=NoSolver(weightsFF, weights=True), seed=seed)
            connBias3 = nengo.Connection(bias, ens3, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)
            connSupv = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            nodeSupv = LearningNode(ens, ens2, 2, conn=connSupv, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, nodeSupv[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, nodeSupv[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(ens3.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(nodeSupv, ens2.neurons, synapse=None)
        if check1:
            pre3 = nengo.Ensemble(nPre, 2, max_rates=m, neuron_type=ReLu(), seed=seed)
            ens3 = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
            nengo.Connection(tarX, pre3, synapse=None, function=feedback)
            connFF2 = nengo.Connection(pre3, ens3, synapse=fTarget, solver=NoSolver(weightsFF, weights=True), seed=seed)
            connBias3 = nengo.Connection(bias, ens3, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)
            assert np.any(weightsFB)
            connSupv = nengo.Connection(ens, ens2, synapse=f1, solver=NoSolver(weightsFB, weights=True))

        if test:
            connFB = nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(weightsFB, weights=True))  # recurrent
            nengo3w = nengo.Connection(tarA, tarA, synapse=fTarget, function=feedback, seed=seed, solver=LstsqL2(weights=True, reg=regNengo))
            

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
        if isinstance(neuron_type, Pyramidal): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)

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
        ens3=sim.data[pEns3] if learn1 or check1 else None,
        tarA=sim.data[pTarA],
        tarA2=sim.data[pTarA2],
        tarX=sim.data[pTarX],
        tarX2=sim.data[pTarX2],
        tarX3=sim.data[pTarX3],
        tarAX=sim.data[pTarAX],
        tarA2X=sim.data[pTarA2X],
        weights1=sim.data[nengo1w].weights.T,
        weights2=sim.data[nengo2w].weights.T,
        weights3=sim.data[nengo3w].weights.T if test else None,
        decoders1=sim.data[nengo1d].weights.T,
        decoders2=sim.data[nengo2d].weights.T,
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

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, tTransTrain=0, tTrans=5, nTrainDF=10, seed=0, rate=60, w=2*np.pi,
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), penalty=0, reg=1e-2, nBins=20,
    load=[]):

    print(f'Neuron type: {neuron_type}')
    rng = np.random.RandomState(seed=seed)

    if 0 in load:
        data = np.load(f"data/oscillateBias_{neuron_type}_{rate}hz.npz")
        dB, eB, wB = data['dB'], data['eB'], data['wB']
    else:
        print('train dB, eB, wB from bias to ens')
        dB, eB, wB = None, None, None
        for n in range(nTrain):
            data = go(neuron_type, learnB=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, rate=rate, w=w,
                dB=dB, eB=eB, wB=wB,
                fTarget=fTarget, fSmooth=fSmooth)
            dB, eB, wB = data['dB'], data['eB'], data['wB']
            np.savez(f"data/oscillateBias_{neuron_type}_{rate}hz.npz",
                dB=dB, eB=eB, wB=wB)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillateBias", neuron_type, "bias", n, nTrain)

        print('check bias to ens connection')
        data = go(neuron_type, checkB=True,
            nEns=nEns, t=10, dt=dt, seed=seed, rate=rate, w=w,
            wB=wB,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "oscillateBias", neuron_type, "bias", -1, 0)


    if 1 in load:
        data = np.load(f"data/oscillateBias_{neuron_type}_{rate}hz.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens (kick input)')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, phase=n/nTrain, w=w)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, rate=rate, w=w,
                wB=wB,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/oscillateBias_{neuron_type}_{rate}hz.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillateBias", neuron_type, "ens", n, nTrain)

        print('check pre to ens connection')
        stim_func1, stim_func2 = makeSignal(10, phase=0.5, w=w)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, check0=True,
            nEns=nEns, t=10, dt=dt, seed=seed, rate=rate, w=w,
            wB=wB,
            w0=w0,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "oscillateBias", neuron_type, "ens", -1, 0)

        print('compare decoder and weight histograms')
        binsNengo = np.linspace(np.min(data['weights1'])/2, np.max(data['weights1'])/2, nBins)
        binsOnline = np.linspace(np.min(w0)/2, np.max(w0)/2, nBins)
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
        sns.histplot(data['weights1'].ravel(), label='nengo', ax=ax, color=palette[0], bins=binsNengo, stat='probability')
        sns.histplot(w0.ravel(), label='online', ax=ax2, color=palette[1], bins=binsOnline, stat='probability')
        ax.set(xlabel='', title='Weights (pre to ens1)')
        ax2.set(xlabel='weight')
        ax.legend()
        ax2.legend()
        plt.tight_layout()
        fig.savefig(f'plots/oscillateBias/weights_pre_to_ens1_{rate}hz.pdf')
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        sns.histplot(data['decoders1'][:,0], label='nengo', ax=ax, color=palette[0], stat='probability')
        sns.histplot(d0[:,0], label='online', ax=ax, color=palette[1], stat='probability')
        ax.set(title='Dimension 0')
        ax.legend()
        sns.histplot(data['decoders1'][:,1], label='nengo', ax=ax2, color=palette[0], stat='probability')
        sns.histplot(d0[:,1], label='online', ax=ax2, color=palette[1], stat='probability')
        ax2.set(xlabel='Decoder', title='Dimension 1')
        ax2.legend()
        plt.tight_layout()
        fig.savefig(f'plots/oscillateBias/decoders_pre_to_ens1_{rate}hz.pdf')

    if 2 in load:
        data = np.load(f"data/oscillateBias_{neuron_type}_{rate}hz.npz")
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('train d1 and f1 for ens to compute the feedback function oscillator')
        # print('train d2 and f2 for ens to readout the representation')
        targets = np.zeros((nTrainDF, int(tTrain/dt), 2))
        targets2 = np.zeros((nTrainDF, int(tTrain/dt), 2))
        spikes = np.zeros((nTrainDF, int(tTrain/dt), nEns))
        for n in range(nTrainDF):
            stim_func1, stim_func2 = makeSignal(tTrain, phase=n/nTrain, w=w)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain+tTransTrain, dt=dt, seed=seed, rate=rate, w=w,
                wB=wB,
                w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            targets[n] = data['tarX2'][int(tTransTrain/dt):]
            targets2[n] = fTarget.filt(data['tarX'],dt=dt)[int(tTransTrain/dt):]
            spikes[n] = data['ens'][int(tTransTrain/dt):]
        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrainDF,
            dt=dt, network="oscillateBias", neuron_type=neuron_type, ens=f"ens",
            tauRiseMin=1e-2, tauRiseMax=3e-2, tauFallMin=8e-2, tauFallMax=2e-1,
            penalty=penalty, seed=seed, reg=reg)
        f1 = DoubleExp(tauRise1, tauFall1)
        np.savez(f"data/oscillateBias_{neuron_type}_{rate}hz.npz",
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
        ax.plot(data['times'], tarAX, linewidth=0.5, label='nengo')
        ax.plot(data['times'], xhat, linewidth=0.5, label='online')
        ax.axhline(1, linewidth=0.5)
        ax.axhline(-1, linewidth=0.5)
        ax.legend()
        ax.set(xlim=((0, tTrain)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/oscillateBias/decode_{neuron_type}_{rate}hz.pdf')

    print(f"taus: {tauRise1:.4f}, {tauFall1:.4f}")

    if 3 in load:
        data = np.load(f"data/oscillateBias_{neuron_type}_{rate}hz.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from ens to ens2')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, phase=n/nTrain, w=w)
            data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt, seed=seed, rate=rate, w=w,
                wB=wB,
                w0=w0,
                d1=d1, f1=f1,
                e1=e1, w1=w1,
                fTarget=fTarget, fSmooth=fSmooth)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/oscillateBias_{neuron_type}_{rate}hz.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
                "oscillateBias", neuron_type, "ens2", n, nTrain)

        print('check ens to ens2 connection')
        stim_func1, stim_func2 = makeSignal(10, phase=0.5, w=w)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, check1=True,
            nEns=nEns, t=10, dt=dt, seed=seed, rate=rate, w=w,
            wB=wB,
            w0=w0,
            d1=d1, f1=f1,
            w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
            "oscillateBias", neuron_type, "ens2", -1, 0)
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
        ax.plot(data['times'], xhat, linewidth=0.5, label='online')
        ax2.plot(data['times'], tarX3, linewidth=0.5, label='target')
        ax2.plot(data['times'], xhat2, linewidth=0.5, label='online')
        ax.axhline(1, linewidth=0.5)
        ax.axhline(-1, linewidth=0.5)
        ax2.axhline(1, linewidth=0.5)
        ax2.axhline(-1, linewidth=0.5)
        ax.set(xlim=((0, 10)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}_1(t))$")
        ax2.set(xlim=((0, 10)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}_2(t))$")
        fig.savefig(f'plots/oscillateBias/feedforward_online_{rate}hz.pdf')
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=((12, 6)), sharex=True)
        ax.plot(data['times'], tarX2, linewidth=0.5, label='target')
        ax.plot(data['times'], tarAX, linewidth=0.5, label='nengo')
        ax2.plot(data['times'], tarX3, linewidth=0.5, label='target')
        ax2.plot(data['times'], tarA2X, linewidth=0.5, label='nengo')
        ax.axhline(1, linewidth=0.5)
        ax.axhline(-1, linewidth=0.5)
        ax2.axhline(1, linewidth=0.5)
        ax2.axhline(-1, linewidth=0.5)
        ax.set(xlim=((0, 10)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}_1(t))$")
        ax2.set(xlim=((0, 10)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}_2(t))$")
        fig.savefig(f'plots/oscillateBias/feedforward_nengo_{rate}hz.pdf')

        print('compare decoder and weight histograms')
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
        sns.histplot(data['weights2'].ravel(), label='nengo', ax=ax, color=palette[0], bins=bins, stat='probability')
        sns.histplot(w1.ravel(), label='online', ax=ax2, color=palette[1], bins=bins, stat='probability')
        ax.set(xlabel='')
        ax2.set(xlabel='Weight')
        ax.legend()
        ax2.legend()
        fig.savefig(f'plots/oscillateBias/weights_ens1_to_ens2_{rate}hz.pdf')
        fig, (ax2, ax3) = plt.subplots(nrows=2, ncols=1, sharex=True)
        sns.histplot(data['decoders2'][:,0], label='nengo', ax=ax2, color=palette[0], stat='probability')
        sns.histplot(d1[:,0], label='online', ax=ax2, color=palette[1], stat='probability')
        ax2.set(xlabel='decoder', title='Dimension 0')
        ax2.legend()
        sns.histplot(data['decoders2'][:,1], label='nengo', ax=ax3, color=palette[0], stat='probability')
        sns.histplot(d1[:,1], label='online', ax=ax3, color=palette[1], stat='probability')
        ax3.set(xlabel='decoder', title='Dimension 1')
        ax3.legend()
        fig.savefig(f'plots/oscillateBias/decoders_ens1_to_ens2_{rate}hz.pdf')

    dfs = []
    columns = ('neuron_type', 'n', 'error rmse', 'error freq', 'dimension')
    print('estimating error')
    for n in range(nTest):
        stim_func1, stim_func2 = makeKick(tTest+tTrans, phase=n/nTest, w=w)
        data = go(neuron_type, stim_func1=stim_func1, stim_func2=stim_func2, test=True,
            nEns=nEns, t=tTest, dt=dt, seed=seed, rate=rate, w=w,
            wB=wB,
            w0=w0,
            f1=f1, d1=d1,
            w1=w1,
            # f2=f2, d2=d2,
            fTarget=fTarget, fSmooth=fSmooth)
        times = data['times']
        preX = data['preX']
        tarX = data['tarX']
        tarX2 = data['tarX2']
        tarAX = data['tarAX']
        aEns = f1.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        aTarA = fSmooth.filt(data['tarA'], dt=dt)
        minRateOnline = np.min(np.max(aEns[int(tTrans/dt):], axis=0))
        maxRateOnline = np.max(aEns[int(tTrans/dt):])
        minRateNengo = np.min(np.max(aTarA[int(tTrans/dt):], axis=0))
        maxRateNengo = np.max(aTarA[int(tTrans/dt):])
        nActiveOnline = np.count_nonzero(np.sum(aEns[int(tTrans/dt):], axis=0))
        nActiveNengo = np.count_nonzero(np.sum(aTarA[int(tTrans/dt):], axis=0))
        print(f'online firing rates: {minRateOnline:.0f} to {maxRateOnline:.0f}Hz')
        print(f'online active neurons: {nActiveOnline}')
        print(f'nengo firing rate range: {minRateNengo:.0f} to {maxRateNengo:.0f}Hz')
        print(f'nengo active neurons: {nActiveNengo}')

        # fig, (ax0, ax, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=((12, 4)))
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((12, 4)))
        # ax0.plot(data['times'], preX, linewidth=0.5, label='pre')
        # ax0.legend(loc='upper right')
        # ax0.set(xlim=((0, tTest)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        # ax.plot(data['times'], tarX2, color='gray', linewidth=0.5, label='target')
        ax.plot(data['times'], xhat, linewidth=0.5, label='online')
        ax.axhline(1, linewidth=0.5)
        ax.axhline(-1, linewidth=0.5)
        ax.legend(loc='upper right')
        ax.set(xlim=((0, tTest)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        # ax2.plot(data['times'], tarX2, color='gray', linewidth=0.5, label='target')
        ax2.plot(data['times'], tarAX, linewidth=0.5, label='nengo')
        ax2.axhline(1, linewidth=0.5)
        ax2.axhline(-1, linewidth=0.5)
        ax2.legend(loc='upper right')
        ax2.set(xlim=((0, tTest)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/oscillateBias/test_compare_{rate}hz.pdf')

        print('compare weight histograms')
        binsNengo = np.linspace(np.min(data['weights3'])/2, np.max(data['weights3'])/2, nBins)
        binsOnline = np.linspace(np.min(w1)/4, np.max(w1)/4, nBins)
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
        sns.histplot(data['weights3'].ravel(), label='nengo', ax=ax, color=palette[0], bins=binsNengo, stat='probability')
        sns.histplot(w1.ravel(), label='online', ax=ax2, color=palette[1], bins=binsOnline, stat='probability')
        ax.set(xlabel='')
        ax2.set(xlabel='Weight')
        ax.legend()
        ax2.legend()
        fig.savefig(f'plots/oscillateBias/weights_recurrent_{rate}hz.pdf')

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
    # fig.savefig('plots/figures/oscillateBias_dim1.pdf')
    # fig.savefig('plots/figures/oscillateBias_dim1.svg')

    # ax2.plot(times, tarX[:,1], label='target, ' + r"$\omega=$" + f"{2*np.pi:.2f}", color='k', linewidth=0.5)
    # ax2.set(xlim=((0, tTest+tTrans)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    # ax2.legend(loc='upper right', frameon=False)
    # plt.tight_layout()
    # fig2.savefig('plots/figures/oscillateBias_dim2.pdf')
    # fig2.savefig('plots/figures/oscillateBias_dim2.svg')

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


compare([LIF()], nTrain=20, nTrainDF=10, eRates=[3e-7], load=[0,1,2,3])