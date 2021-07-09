import numpy as np
import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo import Izhikevich
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengolib import Lowpass, DoubleExp
from utils import LearningNode
from neuron_types import LIF, Wilson, Pyramidal, nrnReset
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')


def makeSignal(t, fIn, dt=0.001, value=1.2, seed=0):
    stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        inpt = nengo.Node(stim)
        probe = nengo.Probe(inpt, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = fIn.filt(sim.data[probe], dt=dt)
    if np.abs(np.max(u)) > np.abs(np.min(u)):
        stim = sim.data[probe] * value / np.max(u)
        if seed%2==0: stim*=-1
    else:
        stim = sim.data[probe] / np.min(u)
        if seed%2==0: stim*=-1
    stim_func = lambda t: stim[int(t/dt)]
    return stim_func

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=1,
    m=Uniform(30, 30), i=Uniform(-0.3, -0.3), e_rate=3e-7, d_rate=1e-6,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1),
    d=None, e=None, w=None, learn=False, stim_func=lambda t: 0):

    weights = w if (np.any(w) and not learn) else np.zeros((nPre, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        tarX = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarA = nengo.Ensemble(nEns, 1, max_rates=m, intercepts=i, encoders=[[1]], neuron_type=ReLu(), seed=seed)
        pre = nengo.Ensemble(nPre, 1, max_rates=m, seed=seed)
        ens = nengo.Ensemble(nEns, 1, gain=[1], bias=[0], encoders=[[1]], neuron_type=neuron_type, seed=seed)

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(inpt, tarX, synapse=fTarget)
        nengo.Connection(pre, tarA, synapse=fTarget)
        conn = nengo.Connection(pre, ens, synapse=fTarget, solver=NoSolver(weights, weights=True))

        if learn:
            node = LearningNode(pre, ens, tarX, 1, conn=conn, d=d, e=e, w=w, e_rate=e_rate, d_rate=d_rate)
            nengo.Connection(pre.neurons, node[:nPre], synapse=fTarget)
            nengo.Connection(ens.neurons, node[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarA.neurons, node[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(inpt, node[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(node, ens.neurons, synapse=None)

        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarA = nengo.Probe(tarA.neurons, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Pyramidal): neuron.h.init()
        sim.run(t, progress_bar=True)
        if isinstance(neuron_type, Pyramidal): nrnReset(sim, model)
    
    if learn:
        d, e, w = node.d, node.e, node.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens=sim.data[pEns],
        tarA=sim.data[pTarA],
        e=e,
        d=d,
        w=w,
    )

def run(neuron_type, nTrain, tTrain, tTest, rate, intercept,
    dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), nBins=21, load=False):

    print(f'Neuron type: {neuron_type}')
    if load:
        data = np.load(f"data/tuning_curve/{neuron_type}.npz")
        d, e, w = data['d'], data['e'], data['w']
    else:
        d, e, w = None, None, None
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, fTarget, dt=dt, seed=n)
            data = go(neuron_type, learn=True,
                t=tTrain, dt=dt, m=Uniform(rate, rate), i=Uniform(intercept, intercept),
                d=d, e=e, w=w,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            d, e, w = data['d'], data['e'], data['w']

            times = data['times']
            tarX = fTarget.filt(data['inpt'], dt=dt)
            aEns = fSmooth.filt(data['ens'], dt=dt)
            aTarA = fSmooth.filt(data['tarA'], dt=dt)
            aPre = fTarget.filt(data['pre'], dt=dt)
            xPre = np.dot(aPre, data['d'])

            fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6, 3)))
            ax.plot(times, tarX, label="input", color='k')
            ax.plot(times, xPre, label="xPre")
            ax.axhline(intercept, color='k', label="intercept", linestyle='--')
            ax.set(xlim=((0, tTrain)), ylim=((-1, 1)), yticks=((-1, 1)), ylabel=r"$\mathbf{x}$(t)")
            ax2.plot(times, aEns, label='ens') 
            ax2.plot(times, aTarA, label='target') 
            ax.legend(loc='upper right', frameon=False)
            ax2.legend(loc='upper right', frameon=False)
            ax2.set(xlim=((0, tTrain)), xticks=((0, tTrain)), ylim=((0, rate)), xlabel='time (s)', ylabel=r"$a(t)$ (Hz)")
            sns.despine()
            plt.tight_layout()
            fig.savefig(f'plots/tuning_curve/train_{neuron_type}_{n+1}outof{nTrain}.pdf')
            plt.close('all')

        np.savez(f"data/tuning_curve/{neuron_type}.npz", d=d, e=e, w=w)

    stim_func = makeSignal(tTest, fTarget, dt=dt, seed=100)
    data = go(neuron_type, learn=False,
        d=d, e=e, w=w,
        t=tTest, dt=dt, m=Uniform(rate, rate), i=Uniform(intercept, intercept),
        fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
    times = data['times']
    tarX = fTarget.filt(data['inpt'], dt=dt)
    aEns = fSmooth.filt(data['ens'], dt=dt)
    aTarA = fSmooth.filt(data['tarA'], dt=dt)

    return times, tarX, aEns, aTarA

def compare(neuron_types, neuron_labels=['LIF', 'Izhikevich', 'Wilson', 'Durstewitz'],
        nTrain=30, tTrain=10, tTest=100, rate=30, intercept=-0.3, nBins=21, tPlot=30, load=False):
    
    bins = np.linspace(-1, 1, nBins)
    activities = []
    binned_activities = []
    mean_activities = []
    CI_activities = []
    for i, neuron_type in enumerate(neuron_types):
        times, tarX, aEns, aTarA = run(neuron_type, nTrain, tTrain, tTest, rate, intercept, load=load)
        activities.append(aEns)
        binned_activities.append([])
        mean_activities.append(np.zeros((nBins, 1)))
        CI_activities.append(np.zeros((2, nBins)))
        for b in range(len(bins)):
            binned_activities[i].append([])
        for t in range(len(times)):
            idx = (np.abs(bins - tarX[t])).argmin()
            binned_activities[i][idx].append(aEns[t][0])
        for b in range(len(bins)):
            mean_activities[i][b] = np.mean(binned_activities[i][b])
            if mean_activities[i][b] > 0:
                CI_activities[i][0][b] = sns.utils.ci(binned_activities[i][b], which=95)[0]
                CI_activities[i][1][b] = sns.utils.ci(binned_activities[i][b], which=95)[1]

    # bin target activities
    neuron_types.append('ReLu()')
    neuron_labels.append('ReLu (target)')
    activities.append(aTarA)
    binned_activities.append([])
    mean_activities.append(np.zeros((nBins, 1)))
    CI_activities.append(np.zeros((2, nBins)))
    for b in range(len(bins)):
        binned_activities[-1].append([])
    for t in range(len(times)):
        idx = (np.abs(bins - tarX[t])).argmin()
        binned_activities[-1][idx].append(aTarA[t][0])
    for b in range(len(bins)):
        mean_activities[-1][b] = np.mean(binned_activities[-1][b])
        if mean_activities[-1][b] > 0:
            CI_activities[-1][0][b] = sns.utils.ci(binned_activities[-1][b], which=95)[0]
            CI_activities[-1][1][b] = sns.utils.ci(binned_activities[-1][b], which=95)[1]

    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6, 4)))
    ax.plot(times, tarX, label="input", color='k')
    ax.axhline(intercept, color='k', linestyle=':')
    ax.set(xlim=((0, tPlot)), ylim=((-1.2, 1.2)), yticks=((-1, 1)), ylabel=r"$\mathbf{x}$(t)")
    for i in range(len(neuron_types)):
        ax2.plot(times, activities[i], label=neuron_labels[i])  # , alpha=0.5
    ax2.legend(loc='upper right', frameon=False)
    ax2.set(xlim=((0, tPlot)), xticks=((0, tPlot)), ylim=((0, rate+5)), yticks=((0, rate)), xlabel='time (s)', ylabel=r"$a(t)$ (Hz)")
    sns.despine(ax=ax, bottom=True)
    sns.despine(ax=ax2)
    plt.tight_layout()
    fig.savefig(f'plots/figures/tuning_curve_activity.pdf')
    fig.savefig(f'plots/figures/tuning_curve_activity.svg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((6, 4)))
    for i in range(len(neuron_types)):
        ax.fill_between(bins, CI_activities[i][0], CI_activities[i][1], alpha=0.1)
        ax.plot(bins, mean_activities[i], label=neuron_labels[i])
    ax.axhline(rate, color='k', linestyle="--", label="target y-intercept")
    ax.axvline(intercept, color='k', linestyle=":", label="target x-intercept")
    ax.set(xlim=((-1, 1)), ylim=((0, rate+5)), xticks=np.array([-1, -0.3, 1]), yticks=((0, rate)), xlabel=r"$\mathbf{x}$", ylabel=r"$a(t)$ (Hz)")
    ax.legend(loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    fig.savefig("plots/figures/tuning_curve_state.pdf")
    fig.savefig("plots/figures/tuning_curve_state.svg")

compare([LIF(), Izhikevich(), Wilson(), Pyramidal()], load=True)