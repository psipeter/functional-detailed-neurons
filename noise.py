import numpy as np

import pandas as pd

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse

from nengolib import Lowpass, DoubleExp

from neuron_types import LIF, Izhikevich, Wilson, NEURON, nrnReset, PoissonSpikingReLU
from utils import LearningNode, trainDF, plotActivities

import neuron

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

# def testPoisson(seed=0):
#     with nengo.Network() as model:
#         ens = nengo.Ensemble(1, 1, encoders=[[1]], intercepts=[-1], max_rates=[50], neuron_type=PoissonSpikingReLU(seed=seed), seed=seed)
#         p = nengo.Probe(ens.neurons, synapse=None)
#     with nengo.Simulator(model) as sim:
#         sim.run(1)
#     fig, ax = plt.subplots()
#     ax.plot(sim.trange(), sim.data[p]/1000)
#     print(np.sum(sim.data[p]/1000))
#     fig.savefig("plots/testPoisson.pdf")
# testPoisson()
# raise


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
        stim = u * value / np.min(u)
        if seed%2==0: stim*=-1
    stim_func = lambda t: stim[int(t/dt)]
    return stim_func

def make_noise(t, nEns, dt=0.001, freq=10, rms=0.3, seed=0):
    return nengo.processes.WhiteSignal(period=t, high=freq, rms=rms, seed=seed)

# def make_noise(t, nEns, dt=0.001, freq=10, rms=0.3, seed=0):
#     noises = [nengo.processes.WhiteSignal(period=t, high=freq, rms=rms, seed=seed+100*n) for n in range(nEns)]
#     with nengo.Network() as model:
#         noises = []
#         inpts = []
#         conns = []
#         inpt = nengo.Node(size_in=nEns)
#         probe = nengo.Probe(inpt, synapse=None)
#         for n in range(nEns):
#             noises.append(nengo.processes.WhiteSignal(period=t, high=freq, rms=rms, seed=seed+100*n))
#             inpts.append(nengo.Node(noises[n]))
#             conns.append(nengo.Connection(inpts[n], inpt[n], synapse=None))
#     with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
#         sim.run(t+dt, progress_bar=False)
#     u = sim.data[probe]
#     noise_func = lambda t: u[int(t/dt)]
#     return noise_func

def go(neuron_type, t=10, seed=0, dt=0.001, nEns=100,
    m=Uniform(20, 40), stim_func=lambda t: 0, noise_func=lambda t: 0,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), f1=DoubleExp(1e-3, 1e-1),
    d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, wN=None, learn0=False, learn1=False, test=False,
    eRate=1e-6, dRate=3e-6):

    weights0 = w0 if (np.any(w0) and not learn0) else np.zeros((nEns, nEns))
    weights1 = w1 if (np.any(w1) and not learn1) else np.zeros((nEns, nEns))
    weightsN = wN if np.any(wN) else np.zeros((nEns, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        noise = nengo.Node(noise_func)
        tarX1 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarX2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarA1 = nengo.Ensemble(nEns, 1, max_rates=m, neuron_type=ReLu(), seed=seed)
        tarA2 = nengo.Ensemble(nEns, 1, max_rates=m, neuron_type=ReLu(), seed=seed+1)
        pre = nengo.Ensemble(nEns, 1, max_rates=m, seed=seed)
        ens1 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(nEns, 1, neuron_type=neuron_type, seed=seed+1)

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarX1, synapse=fTarget)
        nengo.Connection(tarX1, tarX2, synapse=fTarget)
        nengo.Connection(inpt, tarA1, synapse=fTarget)
        nengo.Connection(tarX1, tarA2, synapse=fTarget)
        conn0 = nengo.Connection(pre, ens1, synapse=fTarget, solver=NoSolver(weights0, weights=True))
        conn1 = nengo.Connection(ens1, ens2, synapse=f1, solver=NoSolver(weights1, weights=True))

        if learn0:
            node0 = LearningNode(pre, ens1, 1, conn=conn0, d=d0, e=e0, w=w0, eRate=eRate, dRate=dRate)
            nengo.Connection(pre.neurons, node0[:nEns], synapse=fTarget)
            nengo.Connection(ens1.neurons, node0[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA1.neurons, node0[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(inpt, node0[3*nEns:], synapse=None)
            nengo.Connection(node0, ens1.neurons, synapse=None)
        if learn1:
            node1 = LearningNode(ens1, ens2, 1, conn=conn1, d=d1, e=e1, w=w1, eRate=eRate, dRate=0)
            nengo.Connection(ens1.neurons, node1[:nEns], synapse=f1)
            nengo.Connection(ens2.neurons, node1[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(tarA2.neurons, node1[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(node1, ens2.neurons, synapse=None)
        if test:
            poisson = nengo.Ensemble(nEns, 1, max_rates=m, neuron_type=PoissonSpikingReLU(seed=seed), seed=seed)
            nengo.Connection(noise, poisson, synapse=None)
            connN = nengo.Connection(poisson, ens1, synapse=fTarget, solver=NoSolver(weightsN, weights=True))

        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns1 = nengo.Probe(ens1.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTarA1 = nengo.Probe(tarA1.neurons, synapse=None)
        pTarA2 = nengo.Probe(tarA2.neurons, synapse=None)
        pTarX1 = nengo.Probe(tarX1, synapse=None)
        pTarX2 = nengo.Probe(tarX2, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, NEURON): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, NEURON): nrnReset(sim, model)
    
    if learn0:
        d0, e0, w0 = node0.d, node0.e, node0.w
    if learn1:
        e1, w1 = node1.e, node1.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens1=sim.data[pEns1],
        ens2=sim.data[pEns2],
        tarA1=sim.data[pTarA1],
        tarA2=sim.data[pTarA2],
        tarX1=sim.data[pTarX1],
        tarX2=sim.data[pTarX2],
        e0=e0,
        d0=d0,
        w0=w0,
        e1=e1,
        d1=d1,
        w1=w1,
    )

def run(neuron_type, nTrain, nTest, tTrain, tTest, eRate, noises=[],
    nEns=100, dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), load=[]):

    print(f'Neuron type: {neuron_type}')
    if 0 in load:
        data = np.load(f"data/performance_vs_noise_{neuron_type}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
    else:
        print('train d0, e0, w0 from pre to ens1')
        d0, e0, w0 = None, None, None
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type, learn0=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/performance_vs_noise_{neuron_type}.npz", d0=d0, e0=e0, w0=w0)
            # plotActivities(data['times'], fSmooth.filt(data['ens1'], dt=dt), fSmooth.filt(data['tarA1'], dt=dt),
            #     "noise", neuron_type, "ens1", n, nTrain)

    if 1 in load:
        data = np.load(f"data/performance_vs_noise_{neuron_type}.npz")
        d1, tauRise1, tauFall1 = data['d1'], data['tauRise1'], data['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('train d1 and f1 for ens1 to compute identity')
        targets = np.zeros((nTrain, int(tTrain/dt), 1))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            targets[n] = fTarget.filt(data['tarX1'], dt=dt)
            spikes[n] = data['ens1']

        d1, tauRise1, tauFall1 = trainDF(spikes, targets, nTrain, dt=dt, network="noise", neuron_type=neuron_type, ens="ens1")
        f1 = DoubleExp(tauRise1, tauFall1)
        np.savez(f"data/performance_vs_noise_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)

    if 2 in load:
        data = np.load(f"data/performance_vs_noise_{neuron_type}.npz")
        e1, w1 = data['e1'], data['w1']
        # d1, e1, w1 = data['d1'], data['e1'], data['w1']
    else:
        print('train e1, w1 from ens1 to ens2')
        e1, w1 = None, None
        # d1, e1, w1 = None, None, None
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type, learn1=True, eRate=eRate,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            e1, w1 = data['e1'], data['w1']
            # d1, e1, w1 = data['d1'], data['e1'], data['w1']
            np.savez(f"data/performance_vs_noise_{neuron_type}.npz",
                d0=d0, e0=e0, w0=w0,
                d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
                e1=e1, w1=w1)
            # plotActivities(data['times'], fSmooth.filt(data['ens2'], dt=dt), fSmooth.filt(data['tarA2'], dt=dt),
            #     "noise", neuron_type, "ens2", n, nTrain)

    if 3 in load:
        d2, tauRise2, tauFall2 = data['d2'], data['tauRise2'], data['tauFall2']
        f2 = DoubleExp(tauRise2, tauFall2)
    else:
        print('train d2 and f2 for ens2 for readout')
        targets = np.zeros((nTrain, int(tTrain/dt), 1))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            targets[n] = fTarget.filt(data['tarX2'], dt=dt)
            spikes[n] = data['ens2']

        d2, tauRise2, tauFall2 = trainDF(spikes, targets, nTrain, dt=dt, network="noise", neuron_type=neuron_type, ens="ens2")
        f2 = DoubleExp(tauRise2, tauFall2)
        np.savez(f"data/performance_vs_noise_{neuron_type}.npz",
            d0=d0, e0=e0, w0=w0,
            d1=d1, tauRise1=tauRise1, tauFall1=tauFall1,
            e1=e1, w1=w1,
            d2=d2, tauRise2=tauRise2, tauFall2=tauFall2)

    dfs = []
    columns = ('neuron_type', 'noise', 'n', 'RMSE')
    rng = np.random.RandomState(seed=10)
    for noise in noises:
        for n in range(nTest):
            print(f'noise magnitude {noise}*w0, test {n}')
            wN = noise*np.array(w0)
            rng.shuffle(wN)
            stim_func = makeSignal(tTest, dt=dt, seed=200+n)
            noise_func = make_noise(tTest, nEns, dt=dt, seed=200+n)
            data = go(neuron_type, test=True,
                nEns=nEns, t=tTest, dt=dt,
                d0=d0, e0=e0, w0=w0,
                d1=d1, e1=e1, w1=w1, f1=f1,
                wN=wN,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            times = data['times']
            tarX2 = fTarget.filt(data['tarX2'], dt=dt)
            aEns2 = f2.filt(data['ens2'], dt=dt)
            xhat2 = np.dot(aEns2, d2)
            error2 = rmse(xhat2, tarX2)
            dfs.append(pd.DataFrame([[str(neuron_type)[:-2], noise, n, error2]], columns=columns))

    return dfs

def compare(neuron_types, eRates=[1e-6, 3e-6, 3e-7, 1e-7], nTrain=10, tTrain=10, nTest=5, tTest=10, load=[], noises=[], replot=False):

    if replot:
        data = pd.read_pickle(f"data/performance_vs_noise.pkl")
    else:
        dfs = []
        for i, neuron_type in enumerate(neuron_types):
            df = run(neuron_type, nTrain, nTest, tTrain, tTest, noises=noises, eRate=eRates[i], load=load)
            dfs.extend(df)
            data = pd.concat(dfs, ignore_index=True)
            data.to_pickle(f"data/performance_vs_noise.pkl")
    print(data)

    fig, ax = plt.subplots(figsize=((5.2, 3)))
    sns.lineplot(data=data, x='noise', y='RMSE', hue='neuron_type')
    ax.set(xlabel=r"Magnitude of weighted poisson spikes ($\gamma$)")
    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    fig.savefig('plots/figures/performance_vs_noise.pdf')
    fig.savefig('plots/figures/performance_vs_noise.svg')


compare([LIF(), Izhikevich(), Wilson(), NEURON("Pyramidal")], replot=True, nTest=5, noises=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], load=[0,1,2,3])

def concat():
    data1 = pd.read_pickle(f"data/performance_vs_noise_nonneuron.pkl")
    data2 = pd.read_pickle(f"data/performance_vs_noise_neuron.pkl")
    data = pd.concat([data1, data2], ignore_index=True)
    print(data)
    data.to_pickle(f"data/performance_vs_noise.pkl")
# concat()
