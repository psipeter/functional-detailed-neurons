import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp

from utils import LearningNode, trainDF, trainD, plotActivities
from neuron_types import LIF, Izhikevich, Wilson, NEURON, nrnReset

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
palette = sns.color_palette('dark')
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})



def makeSignal(t, dt=0.001, value=1, seed=0):
    stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        inpt = nengo.Node(stim)
        probe = nengo.Probe(inpt, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = sim.data[probe]
    if np.abs(np.max(u)) > np.abs(np.min(u)):
        stim = u * value / np.max(u)
        if seed%2==0: stim*=-1
    else:
        stim = u / np.min(u)
        if seed%2==0: stim*=-1
    stim_func = lambda t: stim[int(t/dt)]
    return stim_func

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=100,
    m=Uniform(20, 40), eRate=1e-6, dRate=3e-6,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1),
    d=None, e=None, w=None, learn=False, stim_func=lambda t: 0):

    weights = w if (np.any(w) and not learn) else np.zeros((nPre, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        tarX = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarA = nengo.Ensemble(nEns, 1, max_rates=m, neuron_type=ReLu(), seed=seed)
        pre = nengo.Ensemble(nPre, 1, max_rates=m, seed=seed)
        ens = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarX, synapse=fTarget)
        nengo.Connection(pre, tarA, synapse=fTarget)
        conn = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weights, weights=True))

        if learn:
            node = LearningNode(pre, ens, 1, conn=conn, d=d, e=e, w=w, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, node[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, node[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, node[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, node[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(node, ens.neurons, synapse=None)

        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)
        pTarX = nengo.Probe(tarX, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, NEURON): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, NEURON): nrnReset(sim, model)
    
    if learn:
        d, e, w = node.d, node.e, node.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens=sim.data[pEns],
        tarA=sim.data[pTarA],
        tarX=sim.data[pTarX],
        e=e,
        d=d,
        w=w,
    )

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate,
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), load=[]):

    print(f'Neuron type: {neuron_type}')
    if 1 in load:
        data = np.load(f"data/hyperopt_{neuron_type}.npz")
        d, e, w = data['d'], data['e'], data['w']
    else:
        print('train d, e, w from pre to ens')
        d, e, w = None, None, None
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type, learn=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d=d, e=e, w=w,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            d, e, w = data['d'], data['e'], data['w']
            np.savez(f"data/hyperopt_{neuron_type}.npz", d=d, e=e, w=w)
            # plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            #     "hyperopt_stability", neuron_type, "ens", n, nTrain)

    if 2 in load:
        data = np.load(f"data/hyperopt_{neuron_type}.npz")
        rises, falls = data['rises'], data['falls']
    else:
        print('optimizing time constants')
        rises = []
        falls = []
        targets = np.zeros((nTrain, int(tTest/dt), 1))
        spikes = np.zeros((nTrain, int(tTest/dt), nEns))
        for n in range(nTrain):
            stim_func = makeSignal(tTest, value=1.3, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTest, dt=dt,
                d=d, e=e, w=w,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            targets[n] = fTarget.filt(data['tarX'], dt=dt)
            spikes[n] = data['ens']
        for i in range(nTest):
            d, rise, fall = trainDF(spikes, targets, nTrain, seed=i, evals=100,
                dt=dt, network="hyperopt", ens='ens', neuron_type=neuron_type)
            # print(rise, fall)
            rises.append(rise)
            falls.append(fall)
        rises = np.array(rises)
        falls = np.array(falls)
        np.savez(f"data/hyperopt_{neuron_type}.npz", d=d, e=e, w=w, rises=rises, falls=falls)

    return rises, falls

def compare(neuron_types, eRates=[1e-6, 3e-6, 3e-7, 1e-7], nTrain=10, tTrain=10, nTest=100, tTest=10, load=[]):

    dfs = []
    columns = ('neuron_type', 'seed', 'rise', 'fall')
    fig, axes = plt.subplots(nrows=2, ncols=len(neuron_types), figsize=((5.25, 3)), sharey=True)
    for i, neuron_type in enumerate(neuron_types):
        rise, fall = run(neuron_type, nTrain, nTest, tTrain, tTest, eRate=eRates[i], load=load)
        # sns.histplot(rise, stat='percent', ax=axes[0][i], binrange=[0, 0.03], bins=20, color=palette[i])
        # sns.histplot(fall, stat='percent', ax=axes[1][i], binrange=[0.05, 0.2], bins=20, color=palette[i])
        # axes[0][i].set(xlabel=r'$\tau_1$', xlim=((0, 0.03)), xticks=((0, 0.03)), yticks=((0, 100)), ylim=((0, 100)), title=f"{str(neuron_type)[:-2]}")
        # axes[1][i].set(xlabel=r'$\tau_2$', xlim=((0.05, 0.2)), xticks=((0.05, 0.2)), yticks=((0, 100)), ylim=((0, 100)))
        xmin = np.around(np.min(rise), decimals=3)
        xmax = np.around(np.max(rise), decimals=3)
        xlim = ((xmin, xmax))
        sns.histplot(rise, stat='percent', ax=axes[0][i], bins=20, color=palette[i])
        axes[0][i].set(xlabel=r'$\tau_1$', xlim=xlim, xticks=xlim, ylim=((0, 30)), yticks=((0,30)), title=f"{str(neuron_type)[:-2]}")
        xmin = np.around(np.min(fall), decimals=3)
        xmax = np.around(np.max(fall), decimals=3)
        xlim = ((xmin, xmax))
        sns.histplot(fall, stat='percent', ax=axes[1][i], bins=20, color=palette[i])
        axes[1][i].set(xlabel=r'$\tau_2$', xlim=xlim, xticks=xlim,  ylim=((0, 20)), yticks=((0,20)))
    plt.tight_layout()
    fig.savefig("plots/hyperopt/time_constants.svg")

compare([LIF(), Izhikevich(), Wilson(), NEURON('Pyramidal')], load=[1,2])