import numpy as np

import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
# from nengo.utils.ensemble import tunings

from nengolib import Lowpass, DoubleExp
from nengolib.stats import ball, sphere

from utils import LearningNode, plotActivities
from neuron_types import LIF, Izhikevich, Wilson, NEURON, nrnReset

import neuron
import matplotlib.pyplot as plt
import matplotlib.font_manager
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

def go(neuron_type, t=10, seed=0, dt=0.001, nPre=300, nEns=1,
    m=Uniform(30, 30), i=Uniform(-0.3, -0.3), eRate=1e-6, dRate=3e-6,
    fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1),
    d=None, e=None, w=None, learn=False, stim_func=lambda t: 0):

    weights = w if (np.any(w) and not learn) else np.zeros((nPre, nEns))
    with nengo.Network() as model:
        inpt = nengo.Node(stim_func)
        tarX = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tarA = nengo.Ensemble(nEns, 1, max_rates=m, intercepts=i, encoders=[[1]], neuron_type=ReLu(), seed=seed)
        pre = nengo.Ensemble(nPre, 1, max_rates=m, seed=seed)
        ens = nengo.Ensemble(nEns, 1, encoders=[[1]], neuron_type=neuron_type, seed=seed)

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
        e=e,
        d=d,
        w=w,
    )

def run(neuron_type, nTrain, tTrain, tTest, rate, intercept, eRate,
    dt=1e-3, fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-2, 1e-1), nBins=21, load=False):

    print(f'Neuron type: {neuron_type}')
    if load:
        data = np.load(f"data/tuning_{neuron_type}.npz")
        d, e, w = data['d'], data['e'], data['w']
    else:
        d, e, w = None, None, None
        for n in range(nTrain):
            stim_func = makeSignal(tTrain, value=1.3, dt=dt, seed=n)
            data = go(neuron_type, learn=True, eRate=eRate,
                t=tTrain, dt=dt, m=Uniform(rate, rate), i=Uniform(intercept, intercept),
                d=d, e=e, w=w,
                fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
            d, e, w = data['d'], data['e'], data['w']
            # plotActivities(data['times'], fSmooth.filt(data['ens'], dt=dt), fSmooth.filt(data['tarA'], dt=dt),
            #     "tuning", neuron_type, "ens1", n, nTrain)
        np.savez(f"data/tuning_{neuron_type}.npz", d=d, e=e, w=w)

    stim_func = makeSignal(tTest, value=1.3, dt=dt, seed=100)
    data = go(neuron_type, learn=False,
        d=d, e=e, w=w,
        t=tTest, dt=dt, m=Uniform(rate, rate), i=Uniform(intercept, intercept),
        fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
    times = data['times']
    tarX = fTarget.filt(data['inpt'], dt=dt)
    aEns = fSmooth.filt(data['ens'], dt=dt)
    aTarA = fSmooth.filt(data['tarA'], dt=dt)

    return times, tarX, aEns, aTarA

def compare(neuron_types, nTrain=10, tTrain=10, tTest=100, rate=30, intercept=-0.3,
        eRates=[3e-7, 3e-6, 3e-7, 1e-7], nBins=21, tPlot=10, load=False, replot=False):
    
    bins = np.linspace(-1, 1, nBins)
    if replot:
        data = np.load("data/tuning_final.npz")
        times, tarX, activities, CI_activities, mean_activities = data['times'], data['tarX'], data['activities'], data['CI_activities'], data['mean_activities']
        neuron_types.append('ReLu()')
    else:
        activities = []
        binned_activities = []
        mean_activities = []
        CI_activities = []
        for i, neuron_type in enumerate(neuron_types):
            times, tarX, aEns, aTarA = run(neuron_type, nTrain, tTrain, tTest, rate, intercept, eRate=eRates[i], load=load)
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
        np.savez("data/tuning_final.npz",
            times=times, tarX=tarX, activities=activities, CI_activities=CI_activities, mean_activities=mean_activities)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=((5.25, 6)), gridspec_kw={'height_ratios': [1,1,2]})
    axes[0].plot(times, tarX, label="input", color='k')
    axes[0].axhline(intercept, color='k', linestyle=':', linewidth=0.5)
    axes[0].set(xlabel=None, ylabel=r"$\mathbf{x}(t)$", xlim=((0, tPlot)), xticks=(()), ylim=((-1, 1)), yticks=((-1, intercept, 1)))
    for i in range(len(neuron_types)):
        axes[1].plot(times, activities[i], linewidth=0.3)
    axes[1].set(xlabel='time (s)', ylabel=r"$a(t)$", xlim=((0, tPlot)), xticks=((0, tPlot)), ylim=((0, rate+5)), yticks=((0, rate)))
    axes[1].axhline(rate, color='k', linestyle='--', linewidth=0.5)
    for i in range(len(neuron_types)):
        axes[2].fill_between(bins, CI_activities[i][0], CI_activities[i][1], alpha=0.1)
        axes[2].plot(bins, mean_activities[i], label=str(neuron_types[i])[:-2])
    axes[2].axhline(rate, color='k', linestyle="--", label="target y-intercept", linewidth=0.5)
    axes[2].axvline(intercept, color='k', linestyle=":", label="target x-intercept", linewidth=0.5)
    axes[2].set(xlim=((-1, 1)), ylim=((0, rate+5)), xticks=((-1, intercept, 1)), yticks=((0, rate)),
        xlabel=r"$\mathbf{x}$", ylabel=r"$a$ (Hz)")
    axes[2].legend(loc='upper left', frameon=False)
    plt.tight_layout()
    # fig.savefig("plots/figures/tuning_combined_v2.pdf")
    fig.savefig("plots/figures/tuning_combined_v3.svg")
    # fig.savefig('plots/figures/tuning_combined_v2.tiff', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

def ReLuDistribution():
    m = Uniform(20, 40)
    i = Uniform(-1, 1)
    # i = ball.sample(100, 1).T.reshape(-1)
    with nengo.Network() as model:
        ens = nengo.Ensemble(100, 1, max_rates=m, intercepts=i, neuron_type=ReLu())
    with nengo.Simulator(model, progress_bar=False) as sim:
        eval_points, activities = tunings(ens, sim)
    fig, ax = plt.subplots(figsize=((5.25, 2)))
    ax.plot(eval_points, activities, linewidth=0.5)
    ax.set(ylabel=r"Neural Activity $a$ (Hz)", xlabel=r"Input $\mathbf{x}$",
        xlim=((-1, 1)), xticks=((-1, 1)), ylim=((0, 40)), yticks=((0, 40)))
    plt.tight_layout()
    # fig.savefig('plots/figures/ReLuDistribution.pdf')
    fig.savefig('plots/figures/ReLuDistribution.svg')
    # fig.savefig('plots/figures/ReLuDistribution.tiff', dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

def voltageTrace(neuron_type, tTest=1, dt=1e-3,
        fTarget=DoubleExp(1e-3, 1e-1), fSmooth=DoubleExp(1e-3, 1e-1), rate=30, intercept=-0.3):
    data = np.load(f"data/tuning_{neuron_type}.npz")
    d, e, w = data['d'], data['e'], data['w']
    stim_func = lambda t: 1
    data = go(neuron_type, learn=False,
        d=d, e=e, w=w,
        t=tTest, dt=dt, m=Uniform(rate, rate), i=Uniform(intercept, intercept),
        fTarget=fTarget, fSmooth=fSmooth, stim_func=stim_func)
    times = data['times']
    voltage = data['voltage']
    fig, ax = plt.subplots()
    ax.plot(times, voltage)
    ax.set(xlabel='time (s)')
    fig.savefig(f'plots/tuning/{neuron_type}/voltage.pdf')

# ReLuDistribution()
# voltageTrace(NEURON('Pyramidal'))
# compare([NEURON('Interneuron')], eRates=[1e-8], load=False)
# compare([NEURON('Pyramidal')], eRates=[1e-7], load=False)
compare([LIF(), Izhikevich(), Wilson(), NEURON('Pyramidal')], load=False, replot=False)
