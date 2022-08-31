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
# palette = sns.color_palette('dark')
palette = sns.color_palette()
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

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=100, nEns=100,
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
        pV = nengo.Probe(ens.neurons, 'voltage', synapse=None)
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
        voltage=sim.data[pV],
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
        data = np.load(f"data/adaptation_{neuron_type}.npz")
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
            np.savez(f"data/adaptation_{neuron_type}.npz", d=d, e=e, w=w)
            # plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            #     "adaptation", neuron_type, "ens1", n, nTrain)

    if 2 in load:
        data = np.load(f"data/adaptation_{neuron_type}.npz")
        dDefault, dTrain, rise, fall = data['dDefault'], data['dTrain'], data['rise'], data['fall']
        fTrain = DoubleExp(rise, fall)
    else:
        print('train readout decoders and filters')
        targets = np.zeros((nTrain, int(tTrain/dt), 1))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type,
                nEns=nEns, t=tTrain, dt=dt,
                d=d, e=e, w=w,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            targets[n] = fTarget.filt(data['tarX'], dt=dt)
            spikes[n] = data['ens']

        dTrain, rise, fall = trainDF(spikes, targets, nTrain, dt=dt, network="adaptation", ens='ens1', neuron_type=neuron_type)
        fTrain = DoubleExp(rise, fall)
        dDefault = trainD(spikes, targets, nTrain, fTarget, dt=dt)
        np.savez(f"data/adaptation_{neuron_type}.npz",
            d=d, e=e, w=w,
            dDefault=dDefault, dTrain=dTrain, rise=rise, fall=fall)

    dfs = []
    columns = ('neuron_type', 'trial', 'filter', 't', 'tarX', 'xhat', 'error')
    print('estimating error')
    for n in range(nTest):
        stim_func = makeSignal(tTest, dt=dt, seed=100+n)
        data = go(neuron_type,
            nEns=nEns, t=tTest, dt=dt,
            d=d, e=e, w=w,
            fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
        times = data['times']
        tarX = fTarget.filt(data['tarX'], dt=dt)
        aEnsDefault = fTarget.filt(data['ens'], dt=dt)
        aEnsTrain = fTrain.filt(data['ens'], dt=dt)
        xhatDefault = np.dot(aEnsDefault, dDefault)
        xhatTrain = np.dot(aEnsTrain, dTrain)
        for idx, t in enumerate(times):
            dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, 'default', t, tarX[idx,0], xhatDefault[idx,0], np.abs(tarX[idx,0]-xhatDefault[idx,0])]], columns=columns))
            dfs.append(pd.DataFrame([[str(neuron_type)[:-2], n, 'trained', t, tarX[idx,0], xhatTrain[idx,0], np.abs(tarX[idx,0]-xhatTrain[idx,0])]], columns=columns))

    return dfs

def compare(neuron_types, eRates=[1e-6, 3e-6, 3e-7, 1e-7], nTrain=10, tTrain=10, nTest=10, tTest=10, load=[], replot=False):

    if not replot:
        dfs = []
        for i, neuron_type in enumerate(neuron_types):
            df = run(neuron_type, nTrain, nTest, tTrain, tTest, eRate=eRates[i], load=load)
            dfs.extend(df)
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/adaptation.pkl")
    else:
        data = pd.read_pickle(f"data/adaptation.pkl")
    print(data)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=((5.2, 6)), gridspec_kw={'height_ratios': [1,1,2,2]})
    data_tarX = data.query("filter=='default' & trial==0 & neuron_type=='LIF'")
    sns.lineplot(data=data_tarX, x='t', y='tarX', color='k', ax=axes[0], linewidth=0.5)
    for i, neuron_type in enumerate(neuron_types):
        nt = str(neuron_type)[:-2]
        data_xhat = data.query("filter=='default' & trial==0 & neuron_type==@nt")
        sns.lineplot(data=data_xhat, x='t', y='xhat', color=palette[i], ax=axes[0], linewidth=0.5)
    sns.lineplot(data=data_tarX, x='t', y='tarX', color='k', ax=axes[1], linewidth=0.5)
    for i, neuron_type in enumerate(neuron_types):
        nt = str(neuron_type)[:-2]
        data_xhat = data.query("filter=='trained' & trial==0 & neuron_type==@nt")
        sns.lineplot(data=data_xhat, x='t', y='xhat', color=palette[i], ax=axes[1], linewidth=0.5)
    tFilter = 0.3 
    for i, neuron_type in enumerate(neuron_types):
        loaded = np.load(f"data/adaptation_{neuron_type}.npz")
        rise, fall = loaded['rise'], loaded['fall']
        fTrain = DoubleExp(rise, fall)
        nt = str(neuron_type)[:-2]
        if nt=='NEURON': nt='Pyramidal'
        axes[2].plot(fTrain.ntrange(int(tFilter*1000)), fTrain.impulse(int(tFilter*1000)), linewidth=0.5, color=palette[i])
            # label=f"{str(neuron_type)[:-2]}: " + r"$\tau_{\mathrm{rise}}=$"+f"{rise:.2f}s, " + r"$\tau_{\mathrm{fall}}=$"+f"{fall:.2f}s")
    fTarget = DoubleExp(1e-3, 1e-1)
    axes[2].plot(fTarget.ntrange(int(tFilter*1000)), fTarget.impulse(int(tFilter*1000)), linewidth=0.5, color='k')
        # label=r"Default: $\tau_{\mathrm{rise}}=0.001$s, $\tau_{\mathrm{fall}}=0.1$s"))
    sns.barplot(data=data, x='neuron_type', y='error', hue='filter', ax=axes[3])
    axes[0].set(ylim=((-1,1)), yticks=((-1,1)), xlim=((0, tTest)), xticks=(()), xlabel=None, ylabel=r"$\mathbf{\hat{x}}(t)$", title="default filter")
    axes[1].set(ylim=((-1,1)), yticks=((-1,1)), xlim=((0, tTest)), xticks=((0, tTest)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$", title="trained filter")
    axes[2].set(ylim=((0, 10)), yticks=((0,10)), ylabel=r"$h(t)$", xlabel='time (s)', title='filter impulse response', xlim=((0, tFilter)), xticks=((0, tFilter)))
    axes[3].set(ylim=((0, 0.15)), yticks=((0,0.15)), xticklabels=(('LIF', 'Izhikevich', 'Wilson', 'Pyramidal')), ylabel="error", title='improvements with trained filters', xlabel=None)
    axes[3].legend(frameon=False)
    plt.tight_layout()
    fig.savefig("plots/figures/adaptation_combined_v2.svg")

# def print_time_constants():
#     for neuron_type in ['LIF()', 'Izhikevich()', 'Wilson()', 'Pyramidal()']:
#         data = np.load(f"data/adaptation_{neuron_type}.npz")
#         rise, fall = 1000*data['tauRiseOut'], 1000*data['tauFallOut']
#         print(f"{neuron_type}:  \t rise {rise:.3}, fall {fall:.5}")
# print_time_constants()

compare([LIF(), Izhikevich(), Wilson(), NEURON('Pyramidal')], load=[], replot=True)