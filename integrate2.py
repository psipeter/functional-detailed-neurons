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

from neuron_types import LIF, Izhikevich, Wilson, Pyramidal, nrnReset
from utils import LearningNode, trainDF, fitSinusoid
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})


def makeSignal(t, maxX=1, minU=0.8, maxU=1.2, dt=1e-3, seed=0):
    rng = np.random.RandomState(seed=seed)
    done = False
    while not done:
        # stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=rng.randint(1e6))
        stim = nengo.processes.WhiteSignal(period=t/2, high=1.0, rms=1, seed=rng.randint(1e6))
        with nengo.Network() as model:
            inpt = nengo.Node(stim)
            tarX = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            nengo.Connection(inpt, tarX, synapse=1/s)
            pInpt = nengo.Probe(inpt, synapse=None)
            pTarX = nengo.Probe(tarX, synapse=None)
        with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
            # sim.run(t+dt, progress_bar=False)
            sim.run(t/2+dt, progress_bar=False)
        u = sim.data[pInpt]
        x = sim.data[pTarX]
        if np.abs(np.max(x)) > np.abs(np.min(x)):
            inputs = u * maxX / np.max(x)
            targets = x * maxX / np.max(x)
        else:
            inputs = u * maxX / np.min(x)
            targets = x * maxX / np.min(x)
        if seed%2==0 and minU < np.max(np.abs(inputs)) < maxU:
            inputs *= -1
            targets *= -1
            done=True
        if seed%2==1 and minU < np.max(np.abs(inputs)) < maxU:
            done=True
    mirrored_input = np.concatenate((inputs, -inputs), axis=0)
    mirrored_target = np.concatenate((targets, -targets), axis=0)
    # fig, ax = plt.subplots()
    # ax.plot(mirrored_input)
    # ax.plot(mirrored_target)
    # fig.savefig(f'plots/integrate2/signals_{seed}.pdf')
    # raise
    stim_func1 = lambda t: mirrored_input[int(t/dt)]
    stim_func2 = lambda t: mirrored_target[int(t/dt)]
    return stim_func1, stim_func2

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=100,
    m=Uniform(20, 40), i=Uniform(-1, 1), stim_func1=lambda t: 0, stim_func2=lambda t: 0,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), f2=DoubleExp(1e-3, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, d2=None, e2=None, w2=None, wB=None, eB=None, dB=None,
    learnB=False, learn0=False, learn1=False, learn2=False, checkB=False, check0=False, check1=False, check2=False, test=False,
    eRate=1e-6, dRate=3e-6):

    tauRise, tauFall = (-1.0 / np.array(fTarget.poles))
    weightsBias = wB if (np.any(wB) and not learnB) else np.zeros((nPre, nEns))
    weightsX = w0 if (np.any(w0) and not learn0) else np.zeros((nPre, nEns))
    weightsU = w1 if (np.any(w1) and not learn1) else np.zeros((nPre, nEns))
    weightsFB = w2 if (np.any(w2) and not learn2) else np.zeros((nEns, nEns))
    with nengo.Network() as model:
        inptU = nengo.Node(stim_func1)
        inptX = nengo.Node(stim_func2)
        const = nengo.Node(1)
        tarX = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        bias = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)
        preU = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)
        preX = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)
        tarA = nengo.Ensemble(nEns, 1, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)
        ens = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)
        nengo.Connection(inptX, tarX, synapse=None)
        nengo.Connection(inptU, preU, synapse=None)
        nengo.Connection(inptX, preX, synapse=None)
        nengo.Connection(bias, tarA, synapse=fTarget, seed=seed)

        if learnB:
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(np.zeros((nPre, nEns)), weights=True), seed=seed)
            nodeBias = LearningNode(bias, ens, 1, conn=connBias, d=dB, e=eB, w=wB, eRate=eRate, dRate=dRate)
            nengo.Connection(bias.neurons, nodeBias[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeBias[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeBias[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(const, nodeBias[-1:], synapse=fTarget)
            nengo.Connection(nodeBias, ens.neurons, synapse=None)
        if checkB:
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)

        if learn0:  # continue learning bias, but at a low rate
            nengo.Connection(preX, tarA, synapse=fTarget)
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(np.zeros((nPre, nEns)), weights=True), seed=seed)
            nodeBias = LearningNode(bias, ens, 1, conn=connBias, d=dB, e=eB, w=wB, eRate=eRate/3, dRate=dRate)
            nengo.Connection(bias.neurons, nodeBias[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeBias[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeBias[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(const, nodeBias[-1:], synapse=fTarget)
            nengo.Connection(nodeBias, ens.neurons, synapse=None)
            connX = nengo.Connection(preX, ens, synapse=fTarget, solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            nodeX = LearningNode(preX, ens, 1, conn=connX, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(preX.neurons, nodeX[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeX[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeX[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inptX, nodeX[-1], synapse=fTarget)
            nengo.Connection(nodeX, ens.neurons, synapse=None)
        if check0:
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connX = nengo.Connection(preX, ens, synapse=fTarget, solver=NoSolver(weightsX, weights=True))            
            nengo.Connection(preX, tarA, synapse=fTarget)

        if learn1:
            nengo.Connection(preX, tarA, synapse=fTarget)
            nengo.Connection(preU, tarA, synapse=fTarget, transform=tauRise+tauFall)
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connU = nengo.Connection(preU, ens, synapse=fTarget, solver=NoSolver(weightsU, weights=True))
            connX = nengo.Connection(preX, ens, synapse=fTarget, solver=NoSolver(weightsX, weights=True))
            nodeU = LearningNode(preU, ens, 1, conn=connU, d=d1, e=e1, w=w1, eRate=eRate, dRate=dRate)
            nengo.Connection(preU.neurons, nodeU[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeU[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeU[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inptX, nodeU[-1], synapse=fTarget)
            nengo.Connection(inptU, nodeU[-1], synapse=fTarget, transform=tauRise+tauFall)
            nengo.Connection(nodeU, ens.neurons, synapse=None)
        if check1:
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connU = nengo.Connection(preU, ens, synapse=fTarget, solver=NoSolver(weightsU, weights=True))
            connX = nengo.Connection(preX, ens, synapse=fTarget, solver=NoSolver(weightsX, weights=True))
            nengo.Connection(preX, tarA, synapse=fTarget)
            nengo.Connection(preU, tarA, synapse=fTarget, transform=tauRise+tauFall)

        if learn2:
            preU2 = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)  # add one delay before give input to to ens2
            preU3 = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)  # add one delay before give input to ens3
            preX3 = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)  # add one delay before give supervision to ens3
            ens2 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)  # observed activities
            ens3 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)  # target activities
            nengo.Connection(inptU, preU2, synapse=fTarget)
            nengo.Connection(inptU, preU3, synapse=fTarget)
            nengo.Connection(inptX, preX3, synapse=fTarget)
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connBias2 = nengo.Connection(bias, ens2, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connBias3 = nengo.Connection(bias, ens3, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connX = nengo.Connection(preX, ens, synapse=fTarget, solver=NoSolver(weightsX, weights=True))
            connU = nengo.Connection(preU, ens, synapse=fTarget, solver=NoSolver(weightsU, weights=True))
            connU2 = nengo.Connection(preU2, ens2, synapse=fTarget, solver=NoSolver(weightsU, weights=True))  # analogous to connU3
            connX2 = nengo.Connection(ens, ens2, synapse=f2, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))  # analogous to connX3
            connU3 = nengo.Connection(preU3, ens3, synapse=fTarget, solver=NoSolver(weightsU, weights=True))
            connX3 = nengo.Connection(preX3, ens3, synapse=fTarget, solver=NoSolver(weightsX, weights=True))
            nodeX2 = LearningNode(ens, ens2, 1, conn=connX2, d=d2, e=e2, w=w2, eRate=eRate, dRate=0)
            nengo.Connection(ens.neurons, nodeX2[:nEns], synapse=f2)
            nengo.Connection(ens2.neurons, nodeX2[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(ens3.neurons, nodeX2[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(nodeX2, ens2.neurons, synapse=None)
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)
        if check2:
            preU2 = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)  # add one delay before give input to to ens2
            preU3 = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)  # add one delay before give input to ens3
            preX3 = nengo.Ensemble(nPre, 1, max_rates=m, neuron_type=ReLu(), seed=seed)  # add one delay before give supervision to ens3
            ens2 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)  # observed activities
            ens3 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)  # target activities
            nengo.Connection(inptU, preU2, synapse=fTarget)
            nengo.Connection(inptU, preU3, synapse=fTarget)
            nengo.Connection(inptX, preX3, synapse=fTarget)
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connBias2 = nengo.Connection(bias, ens2, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connBias3 = nengo.Connection(bias, ens3, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connX = nengo.Connection(preX, ens, synapse=fTarget, solver=NoSolver(weightsX, weights=True))
            connU = nengo.Connection(preU, ens, synapse=fTarget, solver=NoSolver(weightsU, weights=True))
            connU2 = nengo.Connection(preU2, ens2, synapse=fTarget, solver=NoSolver(weightsU, weights=True))  # analogous to connU3
            connX2 = nengo.Connection(ens, ens2, synapse=f2, solver=NoSolver(w2, weights=True))  # analogous to connX3
            connU3 = nengo.Connection(preU3, ens3, synapse=fTarget, solver=NoSolver(weightsU, weights=True))
            connX3 = nengo.Connection(preX3, ens3, synapse=fTarget, solver=NoSolver(weightsX, weights=True))
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)

        if test:
            connBias = nengo.Connection(bias, ens, synapse=fTarget, solver=NoSolver(weightsBias, weights=True), seed=seed)
            connU = nengo.Connection(preU, ens, synapse=fTarget, solver=NoSolver(weightsU, weights=True))
            connX = nengo.Connection(ens, ens, synapse=f2, solver=NoSolver(weightsFB, weights=True))

        pInptU = nengo.Probe(inptU, synapse=None)
        pInptX = nengo.Probe(inptX, synapse=None)
        pPreU = nengo.Probe(preU.neurons, synapse=None)
        pPreX = nengo.Probe(preX.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarAX = nengo.Probe(tarA, synapse=fTarget)
        pTarX = nengo.Probe(tarX, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Pyramidal): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)
    
    if learnB:
        dB, eB, wB = nodeBias.d, nodeBias.e, nodeBias.w
    if learn0:
        dB, eB, wB = nodeBias.d, nodeBias.e, nodeBias.w
        d0, e0, w0 = nodeX.d, nodeX.e, nodeX.w
    if learn1:
        d1, e1, w1 = nodeU.d, nodeU.e, nodeU.w
    if learn2:
        e2, w2 = nodeX2.e, nodeX2.w

    return dict(
        times=sim.trange(),
        inptU=sim.data[pInptU],
        inptX=sim.data[pInptX],
        preU=sim.data[pPreU],
        preX=sim.data[pPreX],
        ens=sim.data[pEns],
        ens2=sim.data[pEns2] if learn2 or check2 else None,
        ens3=sim.data[pEns3] if learn2 or check2 else None,
        # tarA2=sim.data[pTarA2] if learn2 else None,
        # tarA2X=sim.data[pTarA2X] if learn2 else None,
        tarA=sim.data[pTarA],
        tarAX=sim.data[pTarAX],
        tarX=sim.data[pTarX],
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
        e2=e2,
        d2=d2,
        w2=w2,
        f2=f2,
        dB=dB,
        eB=eB,
        wB=wB,
    )

def run(neuron_type, nTrain, nTrainDF, nTest, tTrain, tTest, eRate,
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1),
    load=[]):

    print(f'Neuron type: {neuron_type}')

    if 0 in load:
        data = np.load(f"data/integrate2_{neuron_type}.npz")
        dB, eB, wB = data['dB'], data['eB'], data['wB']
    else:
        print('train dB, eB, wB from preX to ens')
        dB, eB, wB = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=n)
            data = go(neuron_type, learnB=True, eRate=eRate,
                stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain, dt=dt,
                dB=dB, eB=eB, wB=wB,
                fTarget=fTarget, fSmooth=fSmooth)
            dB, eB, wB = data['dB'], data['eB'], data['wB']
            np.savez(f"data/integrate2_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "integrate2", neuron_type, "bias", n, nTrain)
        print('check bias to ens connection')
        stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=0)
        data = go(neuron_type, checkB=True,
            stim_func1=stim_func1, stim_func2=stim_func2,
            nEns=nEns, t=tTrain, dt=dt,
            wB=wB,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "integrate2", neuron_type, "bias", -1, 0)

    if 1 in load:
        data = np.load(f"data/integrate2_{neuron_type}.npz")
        dB, eB, wB = data['dB'], data['eB'], data['wB']
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from preX to ens')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=n)
            data = go(neuron_type, learn0=True, eRate=eRate,
                stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain, dt=dt,
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            dB, eB, wB = data['dB'], data['eB'], data['wB']
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/integrate2_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "integrate2", neuron_type, "preX", n, nTrain)
        print('check preX to ens connection')
        stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=0)
        data = go(neuron_type, check0=True,
            stim_func1=stim_func1, stim_func2=stim_func2,
            nEns=nEns, t=tTrain, dt=dt,
            wB=wB,
            w0=w0,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "integrate2", neuron_type, "preX", -1, 0)
        print('check bias to ens connection')
        stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=0)
        data = go(neuron_type, check0=True,
            stim_func1=stim_func1, stim_func2=stim_func2,
            nEns=nEns, t=tTrain, dt=dt,
            wB=wB,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "integrate2", neuron_type, "preX", -2, -1)

    if 2 in load:
        data = np.load(f"data/integrate2_{neuron_type}.npz")
        d1, e1, w1 = data['d1'], data['e1'], data['w1']
    else:
        print('train d1, e1, w1 from preU to ens')
        d1, e1, w1 = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=n)
            stim_func2 = lambda tt: 0.5*np.sin(2*np.pi*(tt+n/nTrain)/tTrain)
            data = go(neuron_type, learn1=True, eRate=eRate,
                stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain, dt=dt,
                wB=wB,
                w0=w0,
                d1=d1, e1=e1, w1=w1,
                fTarget=fTarget, fSmooth=fSmooth)
            d1, e1, w1 = data['d1'], data['e1'], data['w1']
            np.savez(f"data/integrate2_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "integrate2", neuron_type, "preU", n, nTrain)
        print('check preU to ens connection')
        stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=0)
        data = go(neuron_type, check1=True, eRate=0,
            stim_func1=stim_func1, stim_func2=stim_func2,
            nEns=nEns, t=tTrain, dt=dt,
            wB=wB,
            w0=w0,
            d1=d1, e1=e1, w1=w1,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "integrate2", neuron_type, "preU", -1, 0)

    if 3 in load:
        d2, tauRise, tauFall = data['d2'], data['tauRise'], data['tauFall']
        f2 = DoubleExp(tauRise, tauFall)
    else:
        print('train d2 and f2 for ens to compute identity for recurrent connection')
        targets = np.zeros((nTrainDF, int(tTrain/dt), 1))
        spikes = np.zeros((nTrainDF, int(tTrain/dt), nEns))
        for n in range(nTrainDF):
            stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=n)
            data = go(neuron_type, check1=True,
                stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain, dt=dt,
                wB=wB,
                w0=w0,
                w1=w1,
                fTarget=fTarget, fSmooth=fSmooth)
            # ens receiving preU and preX, each with fTarget, which should return tarX with no filter. One more filter for ens activities
            targets[n] = fTarget.filt(data['tarX'], dt=dt)
            spikes[n] = data['ens']
        d2, tauRise, tauFall = trainDF(spikes, targets, nTrainDF, dt=dt, network="integrate2", neuron_type=neuron_type, ens="ens")
        f2 = DoubleExp(tauRise, tauFall)
        np.savez(f"data/integrate2_{neuron_type}.npz",
            dB=dB, eB=eB, wB=wB,
            d0=d0, e0=e0, w0=w0,
            d1=d1, e1=e1, w1=w1,
            d2=d2, tauRise=tauRise, tauFall=tauFall)
        times = data['times']
        tarX = fTarget.filt(data['tarX'], dt=dt)
        tarAX = data['tarAX']
        xhat = np.dot(f2.filt(data['ens'], dt=dt), d2)
        fig, ax = plt.subplots(figsize=((5.25, 1.5)))
        ax.plot(times, tarX, linewidth=0.5, label='target')
        # ax.plot(times, tarAX, linewidth=0.5, label='nengo')
        ax.plot(data['times'], xhat, linewidth=0.5, label='xhat')
        ax.set(xlim=((0, tTrain)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/integrate2/decode_{neuron_type}.pdf')
    print(f"taus: {tauRise:.4f}, {tauFall:.4f}")

    if 4 in load:
        data = np.load(f"data/integrate2_{neuron_type}.npz")
        e2, w2 = data['e2'], data['w2']
    else:
        print('train e2, w2 from ens to ens2')
        e2, w2 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=n)
            data = go(neuron_type, learn2=True, eRate=eRate,
                stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain, dt=dt,
                wB=wB,
                w0=w0,
                w1=w1,
                d2=d2, f2=f2,
                e2=e2, w2=w2,
                fTarget=fTarget, fSmooth=fSmooth)
            e2, w2 = data['e2'], data['w2']
            np.savez(f"data/integrate2_{neuron_type}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1,
                d2=d2, tauRise=tauRise, tauFall=tauFall,
                e2=e2, w2=w2)
            # plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
                "integrate2", neuron_type, "ens2", n, nTrain)

        print('check ens to ens2 connection')
        stim_func1, stim_func2 = makeSignal(tTrain, dt=dt, seed=0)
        data = go(neuron_type, check2=True, eRate=0,
            stim_func1=stim_func1, stim_func2=stim_func2,
            nEns=nEns, t=tTrain, dt=dt,
            wB=wB,
            w0=w0,
            w1=w1,
            d2=d2, f2=f2,
            e2=e2, w2=w2,
            fTarget=fTarget, fSmooth=fSmooth)
        # plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
        plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
            "integrate2", neuron_type, "ens2", -1, 0)
        times = data['times']
        tarX = fTarget.filt(data['tarX'], dt=dt)
        tarAX = data['tarAX']
        # tarA2X = data['tarA2X']
        aEns = f2.filt(data['ens'], dt=dt)
        aEns2 = f2.filt(data['ens2'], dt=dt)
        xhat = np.dot(aEns, d2)
        xhat2 = np.dot(aEns2, d2)
        fig, ax = plt.subplots(figsize=((5.25, 1.5)))
        ax.plot(data['times'], tarX, color='k', linewidth=0.5, label='tar')
        # ax.plot(data['times'], tarAX, linewidth=0.5, label='tarA')
        # ax.plot(data['times'], tarA2X, linewidth=0.5, label='tarA2')
        ax.plot(data['times'], xhat, linewidth=0.5, label='xhat')
        ax.plot(data['times'], xhat2, linewidth=0.5, label='xhat2')
        ax.legend()
        ax.set(xlim=((0, tTrain)), xticks=(()), ylim=((-1, 1)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        fig.savefig(f'plots/integrate2/feedforward_{neuron_type}.pdf')

    dfs = []
    columns = ('neuron_type', 'n', 'error')
    print('estimating error')
    for n in range(nTest):
        stim_func1, stim_func2 = makeSignal(tTest, dt=dt, maxX=1.0, seed=100+n)
        data = go(neuron_type,
            nEns=nEns, t=tTest, dt=dt, test=True,
            wB=wB,
            w1=w1,
            w2=w2, f2=f2,
            fTarget=fTarget, fSmooth=fSmooth,
            stim_func1=stim_func1, stim_func2=stim_func2)

        times = data['times']
        inpt = fTarget.filt(data['inptU'], dt=dt)
        tarX = fTarget.filt(data['tarX'], dt=dt)
        aEns = f2.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d2)
        error = rmse(xhat, tarX)

        fig, ax = plt.subplots(figsize=((5.25, 1.5)))
        ax.plot(times, inpt, color='k', linestyle='--', linewidth=0.5)
        ax.plot(times, tarX, color='k', linewidth=0.5)
        ax.plot(times, xhat, linewidth=0.5)
        ax.set(xlim=((0, tTest)), xticks=(()), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
        plt.tight_layout()
        fig.savefig(f'plots/integrate2/integrate2_{neuron_type}_{n}.pdf')
        # fig.savefig(f'plots/integrate2/integrate2_{neuron_type}_{n}.svg')

        dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, error]], columns=columns))

    print(f'observed firing rate range: {np.min(np.max(aEns, axis=0)):.0f} to {np.max(aEns):.0f}Hz')
    # print(f'mean firing rate: {np.mean(aEns, axis=0)}')
    # print(f'active neurons: {np.count_nonzero(np.sum(aEns, axis=0))}/{nEns}')

    return times, inpt, tarX, xhat, dfs

def compare(neuron_types, nTrain=10, nTrainDF=10, tTrain=10, nTest=10, tTest=10, load=[], eRates=[3e-7, 3e-6, 3e-7, 1e-7]):

    dfsAll = []
    fig, ax = plt.subplots(figsize=((5.25, 1.5)))
    for i, neuron_type in enumerate(neuron_types):
        times, inpt, tarX, xhat, dfs = run(neuron_type, nTrain, nTrainDF, nTest, tTrain, tTest, eRate=eRates[i], load=load)
        dfsAll.extend(dfs)
        ax.plot(times, xhat, label=f"{str(neuron_type)[:-2]}", linewidth=0.5)
    df = pd.concat([df for df in dfsAll], ignore_index=True)

    ax.plot(times, inpt, label='input', color='k', linestyle='--', linewidth=0.5)
    ax.plot(times, tarX, label='target', color='k', linewidth=0.5)
    ax.set(xlim=((0, tTest)), xticks=(()), ylim=((-1, 1)), yticks=((-1, 1)), ylabel=r"$\hat{f}(\mathbf{x}(t))$")
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig.savefig('plots/figures/integrate2_ens.pdf')
    fig.savefig('plots/figures/integrate2_ens.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='neuron_type', y='error', ax=ax)
    ax.set(xlabel='', ylim=((0, 0.5)), yticks=((0, 0.5)), ylabel='Error')
    plt.tight_layout()
    fig.savefig('plots/figures/integrate2_barplot.pdf')
    fig.savefig('plots/figures/integrate2_barplot.svg')


compare([LIF()], nTrain=10, load=[])