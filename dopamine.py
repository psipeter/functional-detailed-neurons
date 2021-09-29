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

from neuron_types import LIF, Izhikevich, Wilson, NEURON, nrnReset, AMPA, GABA, NMDA
from utils import LearningNode, trainD, fitSinusoid
from plotter import plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})


def makeSignalCircle(t, dt=0.001, radius=1, rms=0.1, seed=0):
    phase = np.random.RandomState(seed=seed).uniform(0, 1)
    stim = nengo.processes.WhiteSignal(period=t, high=2, rms=rms, seed=seed)
    stim2 = nengo.processes.WhiteSignal(period=t, high=2, rms=rms, seed=50+seed)
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

def goTrain(trainDA, t=10, seed=0, dt=0.001, nPre=300, nEns=100,
    m=Uniform(20, 40), i=Uniform(-0.8, 0.8),
    stim_func1=lambda t: 0, stim_func2=lambda t: 0, stim_func3=lambda t: np.square(np.sin(t)),
    fAMPA=DoubleExp(0.55e-3, 2.2e-3), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fSmooth=DoubleExp(1e-2, 1e-1),
    dB=None, eB=None, wB=None, d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, dI=None, eI=None, wI=None, dL=None, eL=None, wL=None,
    learn0=False, check0=False, learn1=False, check1=False):

    with nengo.Network() as model:
        inpt1 = nengo.Node(stim_func1)
        inpt2 = nengo.Node(stim_func2)
        gate = nengo.Node(stim_func3)
        inpt = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        const = nengo.Node([1,1])
        pre = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        preI = nengo.Ensemble(nPre, 1, max_rates=m, seed=seed)
        bias = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
        inh = nengo.Ensemble(nEns, 1, neuron_type=NEURON('Interneuron', DA=trainDA), seed=seed+1)
        nengo.Connection(inpt1, inpt[0], synapse=None)
        nengo.Connection(inpt2, inpt[1], synapse=None)
        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(gate, preI, synapse=None)

        if learn0:
            tarEns = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)  # feedforward AMPA
            tarEns2 = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)  # feedforward NMDA
            tarInh = nengo.Ensemble(nEns, 1, max_rates=m, intercepts=Uniform(0.5, 1), neuron_type=ReLu(), seed=seed+1)
            ens2 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)

            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)  
            pTarEns2 = nengo.Probe(tarEns2.neurons, synapse=None)  
            pTarInh = nengo.Probe(tarInh.neurons, synapse=None)  
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)

            nengo.Connection(inpt, tarEns, synapse=fAMPA, seed=seed)
            nengo.Connection(bias, tarEns, synapse=fAMPA, seed=seed)  # AMPA
            nengo.Connection(inpt, tarEns2, synapse=fAMPA, seed=seed)
            nengo.Connection(bias, tarEns2, synapse=fNMDA, seed=seed)  # NMDA
            nengo.Connection(gate, tarInh, synapse=fAMPA, seed=seed)

            connB = nengo.Connection(bias, ens, synapse=AMPA(), solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            nodeB = LearningNode(bias, ens, 2, conn=connB, d=dB, e=eB, w=wB, eRate=3e-6, dRate=1e-6)
            nengo.Connection(bias.neurons, nodeB[:nPre], synapse=fAMPA)
            nengo.Connection(ens.neurons, nodeB[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarEns.neurons, nodeB[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(const, nodeB[-2:], synapse=fAMPA)
            nengo.Connection(nodeB, ens.neurons, synapse=None)

            conn0 = nengo.Connection(pre, ens, synapse=AMPA(), solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            node0 = LearningNode(pre, ens, 2, conn=conn0, d=d0, e=e0, w=w0, eRate=3e-6, dRate=1e-6)
            nengo.Connection(pre.neurons, node0[:nPre], synapse=fAMPA)
            nengo.Connection(ens.neurons, node0[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarEns.neurons, node0[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, node0[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(node0, ens.neurons, synapse=None)

            connI = nengo.Connection(preI, inh, synapse=AMPA(), solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            nodeI = LearningNode(preI, inh, 1, conn=connI, d=dI, e=eI, w=wI, eRate=1e-6, dRate=1e-6)
            nengo.Connection(preI.neurons, nodeI[:nPre], synapse=fAMPA)
            nengo.Connection(inh.neurons, nodeI[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarInh.neurons, nodeI[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(gate, nodeI[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(nodeI, inh.neurons, synapse=None)

            connL = nengo.Connection(pre, ens2, synapse=NMDA(), solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            nodeL = LearningNode(pre, ens2, 2, conn=connL, d=dL, e=eL, w=wL, eRate=1e-7, dRate=1e-6)
            nengo.Connection(pre.neurons, nodeL[:nPre], synapse=fNMDA)
            nengo.Connection(ens2.neurons, nodeL[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarEns2.neurons, nodeL[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, nodeL[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(nodeL, ens2.neurons, synapse=None)

            connB2 = nengo.Connection(bias, ens2, synapse=AMPA(), solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            nodeB2 = LearningNode(bias, ens2, 2, conn=connB2, d=dB, e=eB, w=wB, eRate=3e-6, dRate=1e-6)
            nengo.Connection(bias.neurons, nodeB2[:nPre], synapse=fAMPA)
            nengo.Connection(ens.neurons, nodeB2[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarEns.neurons, nodeB2[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(const, nodeB2[-2:], synapse=fAMPA)
            nengo.Connection(nodeB2, ens2.neurons, synapse=None)  # same weight update as connB1, but feeds activity to ens2

        if check0:
            tarEns = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)  # feedforward AMPA
            tarEns2 = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=i, neuron_type=ReLu(), seed=seed)  # feedforward NMDA
            tarInh = nengo.Ensemble(nEns, 1, max_rates=m, intercepts=Uniform(0.5, 1), neuron_type=ReLu(), seed=seed+1)
            ens2 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)

            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)  
            pTarEns2 = nengo.Probe(tarEns2.neurons, synapse=None)  
            pTarInh = nengo.Probe(tarInh.neurons, synapse=None)  
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)

            nengo.Connection(inpt, tarEns, synapse=fAMPA, seed=seed)
            nengo.Connection(inpt, tarEns2, synapse=fAMPA, seed=seed)
            nengo.Connection(bias, tarEns, synapse=fAMPA, seed=seed)
            nengo.Connection(bias, tarEns2, synapse=fAMPA, seed=seed)
            nengo.Connection(gate, tarInh, synapse=fAMPA, seed=seed)
            nengo.Connection(bias, ens, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens2, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(pre, ens, synapse=AMPA(), solver=NoSolver(w0, weights=True))
            nengo.Connection(pre, ens2, synapse=NMDA(), solver=NoSolver(wL, weights=True))
            nengo.Connection(preI, inh, synapse=AMPA(), solver=NoSolver(wI, weights=True))

        if learn1:
            pre2 = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
            ens2 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
            ens3 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
            nengo.Connection(bias, ens, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens2, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens3, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(inpt, pre2, synapse=fAMPA)
            nengo.Connection(pre, ens, synapse=AMPA(), solver=NoSolver(w0, weights=True))
            nengo.Connection(pre2, ens3, synapse=NMDA(), solver=NoSolver(wL, weights=True))
            conn1 = nengo.Connection(ens, ens2, synapse=NMDA(), solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            node1 = LearningNode(ens, ens2, 2, conn=conn1, d=d1, e=e1, w=w1, eRate=1e-7, dRate=0)
            nengo.Connection(ens.neurons, node1[:nEns], synapse=fNMDA)
            nengo.Connection(ens2.neurons, node1[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(ens3.neurons, node1[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(node1, ens2.neurons, synapse=None)
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)

        if check1:
            pre2 = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
            ens2 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
            ens3 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
            nengo.Connection(bias, ens, synapse=fAMPA, solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens2, synapse=fAMPA, solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens3, synapse=fAMPA, solver=NoSolver(wB, weights=True))
            nengo.Connection(inpt, pre2, synapse=fAMPA)
            nengo.Connection(pre, ens, synapse=AMPA(), solver=NoSolver(w0, weights=True))
            nengo.Connection(pre2, ens3, synapse=NMDA(), solver=NoSolver(w0, weights=True))
            nengo.Connection(ens, ens2, synapse=NMDA(), solver=NoSolver(w1, weights=True))
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)            


        pInpt = nengo.Probe(inpt, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pInh = nengo.Probe(inh.neurons, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        neuron.h.init()
        sim.run(t, progress_bar=True)
        nrnReset(sim, model)

    if learn0:
        dB, eB, wB = nodeB.d, nodeB.e, nodeB.w
        dI, eI, wI = nodeI.d, nodeI.e, nodeI.w
        d0, e0, w0 = node0.d, node0.e, node0.w
        dL, eL, wL = nodeL.d, nodeL.e, nodeL.w
    if learn1:
        e1, w1 = node1.e, node1.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        ens=sim.data[pEns],
        inh=sim.data[pInh],
        tarEns=sim.data[pTarEns] if learn0 or check0 else None,
        tarEns2=sim.data[pTarEns2] if learn0 or check0 else None,
        tarInh=sim.data[pTarInh] if learn0 or check0 else None,
        ens2=sim.data[pEns2],
        ens3=sim.data[pEns3] if learn1 or check1 else None,
        eB=eB,
        dB=dB,
        wB=wB,
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
        dI=dI,
        eI=eI,
        wI=wI,
        dL=dL,
        eL=eL,
        wL=wL,
    )

def goTest(testDA, t=10, seed=0, dt=0.001, nPre=300, nEns=100, tGate=1,
    m=Uniform(20, 40), stim_func1=lambda t: 0, stim_func2=lambda t: 0,
    fAMPA=DoubleExp(0.55e-3, 2.2e-3), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3),
    w0=None, w1=None, wB=None, wI=None):

    wInh = -1e-4*np.ones((nEns, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func1)
        gate = nengo.Node(stim_func2)
        bias = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        pre = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        preI = nengo.Ensemble(nPre, 1, max_rates=m, seed=seed)
        diff = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=testDA), seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=testDA), seed=seed)
        inh = nengo.Ensemble(nEns, 1, neuron_type=NEURON('Interneuron', DA=testDA), seed=seed+1)

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(gate, preI, synapse=None)
        nengo.Connection(bias, diff, synapse=AMPA(), solver=NoSolver(wB, weights=True))
        nengo.Connection(bias, ens, synapse=AMPA(), solver=NoSolver(wB, weights=True))
        nengo.Connection(pre, diff, synapse=AMPA(), solver=NoSolver(w0, weights=True))
        nengo.Connection(preI, inh, synapse=AMPA(), solver=NoSolver(wI, weights=True))
        nengo.Connection(diff, ens, synapse=NMDA(), solver=NoSolver(w1, weights=True))
        nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(w1, weights=True))
        nengo.Connection(ens, diff, synapse=NMDA(), solver=NoSolver(-w1, weights=True))
        # nengo.Connection(inh, diff, synapse=GABA(), solver=NoSolver(wInh, weights=True))

        pInpt = nengo.Probe(inpt, synapse=None)
        pGate = nengo.Probe(gate, synapse=None)
        pDiff = nengo.Probe(diff.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pInh = nengo.Probe(inh.neurons, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        neuron.h.init()
        sim.run(tGate+t, progress_bar=True)
        nrnReset(sim, model)

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        gate=sim.data[pGate],
        diff=sim.data[pDiff],
        ens=sim.data[pEns],
        inh=sim.data[pInh],
    )


def train(trainDA, seed, load=[],
    nTrain=10, tTrain=10, tGate=1, nEns=10, dt=1e-3,
    fAMPA=DoubleExp(0.55e-3, 2.2e-3), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fSmooth=DoubleExp(1e-2, 1e-1)):

    if 0 in load:
        data = np.load(f"data/dopamine_trainDA{trainDA}_seed{seed}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
        dB, eB, wB = data['dB'], data['eB'], data['wB']
        dI, eI, wI = data['dI'], data['eI'], data['wI']
        dL, eL, wL = data['dL'], data['eL'], data['wL']
    else:
        print('train d0, e0, w0 from pre to diff')
        d0, e0, w0 = None, None, None
        dB, eB, wB = None, None, None
        dI, eI, wI = None, None, None
        dL, eL, wL = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=n)
            data = goTrain(trainDA=trainDA, learn0=True,
                stim_func1=stim_func1, stim_func2=stim_func2,
                nEns=nEns, t=tTrain, dt=dt, seed=seed,
                dB=dB, eB=eB, wB=wB,
                dI=dI, eI=eI, wI=wI,
                dL=dL, eL=eL, wL=wL,
                d0=d0, e0=e0, w0=w0)
            dB, eB, wB = data['dB'], data['eB'], data['wB']
            dI, eI, wI = data['dI'], data['eI'], data['wI']
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            dL, eL, wL = data['dL'], data['eL'], data['wL']
            np.savez(f"data/dopamine_trainDA{trainDA}_seed{seed}.npz",
                dB=dB, eB=eB, wB=wB,
                dI=dI, eI=eI, wI=wI,
                dL=dL, eL=eL, wL=wL,
                d0=d0, e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarEns'], dt=dt),
                "dopamine", 'NEURON', "pre", n, nTrain)
            plotActivities(data['times'], fSmooth.filt(data['inh'], dt=dt), fSmooth.filt(data['tarInh'], dt=dt),
                "dopamine", 'NEURON', "inh", n, nTrain)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarEns2'], dt=dt),
                "dopamine", 'NEURON', "pre2", n, nTrain)
        stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=0)
        data = goTrain(trainDA=trainDA, check0=True,
            nEns=nEns, t=tTrain, dt=dt, seed=seed,
            stim_func1=stim_func1, stim_func2=stim_func2,
            wB=wB,
            wI=wI,
            wL=wL,
            w0=w0)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarEns'], dt=dt),
            "dopamine", 'NEURON', "pre", -1, 0)
        plotActivities(data['times'], fSmooth.filt(data['inh'], dt=dt), fSmooth.filt(data['tarInh'], dt=dt),
            "dopamine", 'NEURON', "inh", -1, 0)
        plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarEns2'], dt=dt),
            "dopamine", 'NEURON', "pre2", -1, 0)

    if 1 in load:
        data = np.load(f"data/dopamine_trainDA{trainDA}_seed{seed}.npz")
        d1 = data['d1']
        d2 = data['d2']
    else:
        print('train d1 for diff/ens to compute identity')
        targets = np.zeros((nTrain, int(tTrain/dt), 2))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=n)
            data = goTrain(trainDA=trainDA, check0=True,
                nEns=nEns, t=tTrain, dt=dt, seed=seed,
                stim_func1=stim_func1, stim_func2=stim_func2,
                wB=wB,
                w0=w0,
                wL=wL,
                wI=wI)
            targets[n] = fNMDA.filt(fAMPA.filt(data['inpt'], dt=dt), dt=dt)
            spikes[n] = data['ens']
        d1 = trainD(spikes, targets, nTrain, fNMDA, dt=dt)
        d2 = -d1
        np.savez(f"data/dopamine_trainDA{trainDA}_seed{seed}.npz",
            dB=dB, eB=eB, wB=wB,
            d0=d0, e0=e0, w0=w0,
            dI=dI, eI=eI, wI=wI,
            dL=dL, eL=eL, wL=wL,
            d1=d1, d2=d2)
        times = data['times']
        inpt = data['inpt']
        target = fNMDA.filt(fAMPA.filt(data['inpt'], dt=dt), dt=dt)
        aEns = fNMDA.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        fig, ax = plt.subplots()
        ax.plot(times, target, label='target')
        ax.plot(times, xhat, label='xhat')
        ax.legend()
        ax.set(yticks=((-1,1)))
        fig.savefig(f'plots/dopamine/decode.pdf')

    if 2 in load:
        data = np.load(f"data/dopamine_trainDA{trainDA}_seed{seed}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from diff/ens to ens')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=n)
            data = goTrain(trainDA=trainDA, learn1=True,
                nEns=nEns, t=tTrain, dt=dt, seed=seed,
                stim_func1=stim_func1, stim_func2=stim_func2,
                wB=wB,
                w0=w0,
                wI=wI,
                wL=wL,
                d1=d1, e1=e1, w1=w1)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/dopamine_trainDA{trainDA}_seed{seed}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                dI=dI, eI=eI, wI=wI,
                dL=dL, eL=eL, wL=wL,
                d1=d1, d2=d2,
                e1=e1, w1=w1)
            plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
                "dopamine", 'NEURON', "ens", n, nTrain)
        stim_func1, stim_func2 = makeSignalCircle(tTrain, dt=dt, seed=0)
        data = goTrain(trainDA=trainDA, check1=True,
            nEns=nEns, t=tTrain, dt=dt, seed=seed,
            stim_func1=stim_func1, stim_func2=stim_func2,
            wB=wB,
            w0=w0,
            wL=wL,
            w1=w1)
        plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['ens3'], dt=dt),
            "dopamine", 'NEURON', "ens", -1, 0)

def test(trainDA, testDA, seed,
    nTest=8, tTest=8, tGate=1, nEns=100, dt=1e-3,
    fAMPA=DoubleExp(0.55e-3, 2.2e-3), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3)):

    data = np.load(f"data/dopamine_trainDA{trainDA}_seed{seed}.npz")
    wB, wI, w0, w1, d1 = data['wB'], data['wI'], data['w0'], data['w1'], data['d1']

    dfs = []
    columns = ('trainDA', 'testDA', 'seed', 'trial', 'error')
    print('estimating error')
    angles = np.linspace(0, 2*np.pi, nTest+1)
    for n in range(nTest):
        stim_func1, stim_func2 = makeTest(tTest, angle=angles[n], tGate=tGate)
        data = goTest(testDA=testDA,
            nEns=nEns, t=tTest, tGate=tGate, dt=dt, seed=seed,
            wB=wB,
            wI=wI,
            w0=w0,
            w1=w1,
            stim_func1=stim_func1, stim_func2=stim_func2)
        times = data['times']
        target = data['inpt']
        aEns = fNMDA.filt(data['ens'], dt=dt)
        xhat = np.dot(aEns, d1)
        error = rmse(xhat, target)
        dfs.append(pd.DataFrame([[trainDA, testDA, seed, n, error]], columns=columns))
        
        fig, ax, = plt.subplots()
        ax.plot(times, target, label='target')
        ax.plot(times, xhat, label='xhat')
        ax.legend()
        ax.set(yticks=((-1,1)), xticks=((0, tGate, tGate+tTest)))
        fig.savefig(f'plots/dopamine/time_trainDA{trainDA}_testDA{testDA}_seed{seed}_trial{n}.pdf')

        fig, ax = plt.subplots()
        ax.plot(xhat[int(tGate/dt):,0], xhat[int(tGate/dt):,1], label='xhat, rmse=%.3f'%error, zorder=1)
        ax.scatter(target[0,0], target[0,1], s=8, color='k', label='target', zorder=2)
        ax.legend()
        ax.set(yticks=((-1,1)), xticks=((-1,1)))
        fig.savefig(f'plots/dopamine/space_trainDA{trainDA}_testDA{testDA}_seed{seed}_trial{n}.pdf')
        plt.close('all')

    return dfs


def compare(seeds=[0], trainDAs=[0], testDAs=[0,1], load=[]):
    dfs = []
    columns = ('trainDA', 'testDA', 'seed', 'trial', 'error')
    for seed in seeds:
        for trainDA in trainDAs:
            train(trainDA, seed=seed, load=load)
            for testDA in testDAs:
                dfs.extend(test(trainDA, testDA, seed=seed))
    df = pd.concat([df for df in dfs], ignore_index=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((5.25, 1.5)))
    sns.barplot(data=df, x='testDA', y='error', hue='trainDA', ax=ax)
    ax.set(ylabel='Error')
    plt.tight_layout()
    fig.savefig('plots/figures/dopamine_barplot.pdf')
    fig.savefig('plots/figures/dopamine_barplot.svg')

compare(load=[0,1])