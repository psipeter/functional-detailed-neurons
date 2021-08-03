import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver, Lstsq, LstsqL2
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

def go(neuron_type, t=10, seed=0, dt=1e-3, nPre=300, nEns=100, m=Uniform(20, 40),
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, learn0=False, learn1=False, test=False,
    eRate=1e-6, dRate=3e-6, tKick=1):

    def feedback(x):
        tau = 0.1
        w = 2*np.pi
        r = np.maximum(np.sqrt(x[0]**2 + x[1]**2), 1e-9)
        dx0 = x[0]*(1-r**2)/r - x[1]*w 
        dx1 = x[1]*(1-r**2)/r + x[0]*w 
        return [tau*dx0 + x[0],  tau*dx1 + x[1]]

    d0 = d0 if np.any(d0) else np.zeros((nEns, 2))
    d1 = d1 if np.any(d1) else np.zeros((nEns, 2))
    with nengo.Network() as model:
        inpt = nengo.Node(lambda t: [1, 0] if t<0.1 else [0,0])  # square wave kick
        tarA = nengo.Ensemble(nEns, 2, gain=Uniform(1.2, 2.0), bias=Uniform(0,0), neuron_type=nengo.LIF(), seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=neuron_type, seed=seed)
        xhatTarA = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        xhatEns = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())

        nengo.Connection(inpt, tarA, synapse=fTarget)
        connTarFB = nengo.Connection(tarA, tarA, synapse=fTarget, function=feedback)
        connTarOut = nengo.Connection(tarA, xhatTarA, synapse=fTarget, function=feedback)
        connEnsOut = nengo.Connection(ens, xhatEns, synapse=fTarget, solver=NoSolver(d1, weights=False))

        if learn1:
            connSupv = nengo.Connection(tarA, ens, synapse=fTarget, solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            nodeSupv = LearningNode(tarA, ens, 2, conn=connSupv, d=d0, e=e0, w=w0, eRate=eRate, dRate=0)
            nengo.Connection(tarA.neurons, nodeSupv[:nEns], synapse=fTarget)
            nengo.Connection(ens.neurons, nodeSupv[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, nodeSupv[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(xhatTarA, nodeSupv[-2:], synapse=None)
            nengo.Connection(nodeSupv, ens.neurons, synapse=None)

        if test:
            connSupv = nengo.Connection(tarA, ens, synapse=fTarget, solver=NoSolver(w0, weights=True))  # kick
            connEnsFB = nengo.Connection(ens, ens, synapse=fTarget, solver=NoSolver(w0, weights=True))  # recurrent
            off = nengo.Node(lambda t: 0 if t<=tKick else -1e4)  # remove kick
            nengo.Connection(off, tarA.neurons, synapse=None, transform=np.ones((nEns, 1)))

        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pXhatTarA = nengo.Probe(xhatTarA, synapse=None)
        pXhatEns = nengo.Probe(xhatEns, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        sim.run(t, progress_bar=True)

    if learn0:
        d0 = sim.data[connTarFB].weights
        d1 = sim.data[connTarOut].weights
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
        e1=e1,
        d1=d1,
        w1=w1,
    )

def run(neuron_type, nTrain, tTrain, tTest, eRate,
    nEns=500, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1),
    load=[]):

    if 0 in load:
        data = np.load(f"data/oscillateAttractor_{neuron_type}.npz")
        d0, d1 = data['d0'], data['d1']
    else:
        print('decoders for tarA')
        data = go(neuron_type, nEns=nEns, t=tTrain, fTarget=fTarget, learn0=True)
        d0, d1 = data['d0'].T, data['d1'].T
        np.savez(f"data/oscillateAttractor_{neuron_type}.npz",
            d0=d0, d1=d1)

        fig, ax = plt.subplots()
        ax.plot(data['times'], data['xhatTarA'], label='xhat (ReLU)')
        ax.plot(data['times'], data['xhatEns'], label='xhat (ens)')
        ax.legend()
        fig.savefig('plots/oscillate/attractor_tarA.pdf')

    if 1 in load:
        data = np.load(f"data/oscillateAttractor_{neuron_type}.npz")
        e0, w0 = data['e0'], data['w0']
    else:
        print('encoders/weights for supv')
        e0, w0 = None, None
        for n in range(nTrain):
            data = go(neuron_type, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, d1=d1,
                e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth)
            e0, w0 = data['e0'], data['w0']
            np.savez(f"data/oscillate_{neuron_type}.npz",
                d0=d0, d1=d1,
                e0=e0, w0=w0)
            plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
                "oscillate", neuron_type, "ens", n, nTrain)

    if 2 in load:
        print('checking activities/estimate with zero learning rate')
        data = go(neuron_type, learn1=True, eRate=0,
            nEns=nEns, t=tTrain, dt=dt,
            d0=d0, d1=d1,
            e0=e0, w0=w0,
            fTarget=fTarget, fSmooth=fSmooth)
        plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            "oscillate", neuron_type, "ens", -1, 0)

        fig, ax = plt.subplots()
        ax.plot(data['times'], data['xhatTarA'], label='xhat (ReLU)')
        ax.plot(data['times'], data['xhatEns'], label='xhat (ens)')
        ax.legend()
        fig.savefig('plots/oscillate/attractor_supv.pdf')

    print('testing')
    data = go(neuron_type, test=True,
        nEns=nEns, t=tTest, dt=dt,
        d0=d0, d1=d1,
        w0=w0,
        fTarget=fTarget)
    fig, ax = plt.subplots()
    ax.plot(data['times'], data['xhatTarA'], label='xhat (ReLU)')
    ax.plot(data['times'], data['xhatEns'], label='xhat (ens)')
    ax.legend()
    fig.savefig('plots/oscillate/attractor_test.pdf')

run(LIF(), nTrain=10, tTrain=20, tTest=10, eRate=3e-7, load=[2])