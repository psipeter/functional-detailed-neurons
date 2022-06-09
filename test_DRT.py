import numpy as np
import pandas as pd
import nengo
from nengo import SpikingRectifiedLinear as ReLu
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.stats import sphere
from nengolib.synapses import ss2sim
from nengolib.signal import LinearSystem, s
from neuron_types import LIF, Izhikevich, Wilson, NEURON, nrnReset, AMPA, GABA, NMDA
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

class CueInput():
    def __init__(self):
        self.cue = [0,0]
    def set(self, cue):
        self.cue = cue
    def get(self):
        return self.cue

class GateInput():
    def __init__(self, tGate):
        self.tGate = tGate
    def get(self, t):
        return [0,0] if t<self.tGate else [1,1]

def baseline_network(seed, cueInpt, gateInpt, cues, nPre=1000, nEns=100):

    m = Uniform(20, 40)
    fAMPA = DoubleExp(0.55e-3, 2.2e-3)
    fNMDA = DoubleExp(10.6e-3, 285e-3)
    fGABA = DoubleExp(0.5e-3, 1.5e-3)

    eInh = Choice([[1,1]])
    ePre = sphere.sample(nPre, 2, rng=np.random.RandomState(seed=seed))
    eTarget = sphere.sample(nEns, 2, rng=np.random.RandomState(seed=seed))
    wInh = -1e-1*np.ones((nEns, nEns))

    network = nengo.Network(seed=seed)
    with network:    
        inpt = nengo.Node(lambda t, x: cueInpt.get(), size_in=2, size_out=2)
        gate = nengo.Node(lambda t, x: gateInpt.get(t), size_in=2, size_out=2)
        pre = nengo.Ensemble(nPre, 2, radius=1.4, max_rates=m, encoders=ePre, seed=seed)
        preI = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        diff = nengo.Ensemble(nEns, 2, radius=1.4, max_rates=m, encoders=eTarget, seed=seed)
        ens = nengo.Ensemble(nEns, 2, radius=1.4, max_rates=m, encoders=eTarget, seed=seed)
        inh = nengo.Ensemble(nEns, 2, radius=1.4, max_rates=m, intercepts=Uniform(0.5, 1), encoders=eInh, seed=seed+1)
        cleanup = nengo.networks.AssociativeMemory(cues, n_neurons=nEns, seed=seed)
        cleanup.add_wta_network(inhibit_synapse=fGABA)
        for pop in cleanup.am_ensembles:
            pop.neuron_type = nengo.LIF()
            pop.max_rates = m

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(gate, preI, synapse=None)
        nengo.Connection(pre, diff, synapse=fAMPA)
        nengo.Connection(preI, inh, synapse=fAMPA)
        nengo.Connection(diff, ens, synapse=fNMDA)
        nengo.Connection(ens, ens, synapse=fNMDA)
        nengo.Connection(ens, diff, synapse=fNMDA, transform=-1)
        nengo.Connection(inh.neurons, diff.neurons, synapse=fGABA, transform=wInh)
        nengo.Connection(ens, cleanup.input, synapse=fNMDA)

        network.pInpt = nengo.Probe(inpt, synapse=None)
        network.pGate = nengo.Probe(gate, synapse=None)
        network.pDiff = nengo.Probe(diff, synapse=fNMDA)
        network.pEns = nengo.Probe(ens, synapse=fNMDA)
        network.pInh = nengo.Probe(inh, synapse=fNMDA)
        network.pCleanup = nengo.Probe(cleanup.output, synapse=fNMDA)

    return network

def baseline_test(network, t=8, tGate=2, dt=0.001):
    sim = nengo.Simulator(network, dt=dt, progress_bar=False)
    sim.run(tGate+t, progress_bar=True)
    data = dict(
        times=sim.trange(),
        inpt=sim.data[network.pInpt],
        gate=sim.data[network.pGate],
        diff=sim.data[network.pDiff],
        ens=sim.data[network.pEns],
        inh=sim.data[network.pInh],
        cleanup=sim.data[network.pCleanup])
    del(sim)
    return data

def run_baseline(nTest=16, nSeeds=10, t=20, tGate=1, plot=True, load=False, thr=0.1):
    columns = ('seed', 'trial', 'delay length', 'error', 'error_cleanup', 'correct')
    dfs = [] 
    if load=='pkl':
        data = pd.read_pickle(f"data/delayed_response_task_baseline.pkl")
    elif load=='npz':
        dfs = []
        for seed in range(nSeeds):
            print(f"seed {seed}")
            for n in range(nTest): 
                df = np.load(f"data/delayed_response_task_baseline_seed{seed}_trial{n}.npz")
                for i in range(len(df['times'])):
                    dfs.append(pd.DataFrame([[
                        seed, n, df['times'][i]-tGate, df['error_estimate'][i], df['error_cleanup'][i], df['correct'][i]]], columns=columns))
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/delayed_response_task_baseline.pkl")
    else:
        angles = np.linspace(0, 2*np.pi, nTest+1)[:-1]
        cues = np.array([[np.sin(angle), np.cos(angle)] for angle in angles])
        cueInpt = CueInput()
        gateInpt = GateInput(tGate)
        for seed in range(nSeeds):
            network = baseline_network(seed, cueInpt, gateInpt, cues)
            for n in range(nTest):
                cueInpt.set(cues[n])
                data = baseline_test(network, t=t, tGate=tGate)
                error_estimate = np.sqrt(np.square(data['inpt'][:,0]-data['ens'][:,0]) + np.square(data['inpt'][:,1]-data['ens'][:,1]))
                error_cleanup = np.sqrt(np.square(data['inpt'][:,0]-data['cleanup'][:,0]) + np.square(data['inpt'][:,1]-data['cleanup'][:,1]))
                correct = np.array([100 if error_cleanup[i]<thr else 0 for i in range(len(data['times']))])
                np.savez(f"data/delayed_response_task_baseline_seed{seed}_trial{n}.npz",
                    times=data['times'],
                    inpt=data['inpt'],
                    ens=data['ens'],
                    cleanup=data['cleanup'],
                    error_estimate=error_estimate,
                    error_cleanup=error_cleanup,
                    correct=correct)
                if plot:
                    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6,4)))
                    ax.plot(data['times'], data['inpt'], label='target')
                    ax.plot(data['times'], data['ens'], label='ens')
                    ax.plot(data['times'], data['cleanup'], label='cleanup')
                    ax2.plot(data['times'], correct)
                    # ax.plot(data['times'], error_estimate, label='error estimate')
                    # ax.plot(data['times'], error_cleanup, label='error cleanup')
                    ax.set(ylabel='estimate')
                    ax.legend()
                    ax2.set(xlabel='time (s)', ylabel='correct')
                    fig.savefig(f'plots/delayed_response_task/baseline_seed{seed}_trial{n}.pdf')
                    plt.close('all')
        dfs = []
        for seed in range(nSeeds):
            print(f"seed {seed}")
            for n in range(nTest): 
                data = np.load(f"data/delayed_response_task_baseline_seed{seed}_trial{n}.npz")
                for i in range(len(data['times'])):
                    dfs.append(pd.DataFrame([[
                        seed, n, data['times'][i]-tGate, data['error_estimate'][i], data['error_cleanup'][i], data['correct'][i]]], columns=columns))
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/delayed_response_task_baseline.pkl")

    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6,4)))
    sns.lineplot(data=data, x='delay length', y='error', ax=ax)
    sns.lineplot(data=data, x='delay length', y='correct', ax=ax2)
    ax.set(xlim=((0, t)), xticks=((0, t)), ylim=((0, 0.4)), yticks=((0, 1)), ylabel='Error (Euclidean)')
    ax2.set(xlim=((0, t)), xticks=((0, t)), ylim=((0, 100)), yticks=((0, 100)), ylabel='Percent Correct', xlabel='Delay Length (s)')
    plt.tight_layout()
    fig.savefig(f'plots/delayed_response_task/baseline.pdf')

# run_baseline(load='pkl')



def bioneuron_test(cueInpt, gateInpt, cues, t, dt, tGate,
    trainDA=0.0, testDA=0.0, seed=0, nPre=1000, nEns=100, m=Uniform(20, 40),
    fAMPA=DoubleExp(0.55e-3, 2.2e-3), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3),
    w0=None, w1=None, wB=None, wI=None, w2=None):

    weights = np.load(f"data/dopamine_trainDA{trainDA}_seed{seed}.npz")
    wB, wI, w0, w1, d1, w2 = weights['wB'], weights['wI'], weights['w0'], weights['w1'], weights['d1'], weights['w2']
    w2 = -w1
    ePre = sphere.sample(nPre, 2, rng=np.random.RandomState(seed=seed))
    wInh = -1e-1*np.ones((nEns, nEns))
    with nengo.Network() as network:
        inpt = nengo.Node(lambda t, x: cueInpt.get(), size_in=2, size_out=2)
        gate = nengo.Node(lambda t, x: gateInpt.get(t), size_in=2, size_out=2)
        bias = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        pre = nengo.Ensemble(nPre, 2, radius=1.4, max_rates=m, encoders=ePre, seed=seed)
        preI = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        diff = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=testDA), seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=testDA), seed=seed)
        inh = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Interneuron', DA=testDA), seed=seed+1)
        cleanup = nengo.networks.AssociativeMemory(cues, n_neurons=nEns, seed=seed)
        cleanup.add_wta_network(inhibit_synapse=fGABA)
        for pop in cleanup.am_ensembles:
            pop.neuron_type = nengo.LIF()
            pop.max_rates = m

        nengo.Connection(inpt, pre, synapse=None)
        nengo.Connection(gate, preI, synapse=None)
        nengo.Connection(bias, diff, synapse=AMPA(), solver=NoSolver(wB, weights=True))
        nengo.Connection(bias, ens, synapse=AMPA(), solver=NoSolver(wB, weights=True))
        nengo.Connection(pre, diff, synapse=AMPA(), solver=NoSolver(w0, weights=True))
        nengo.Connection(preI, inh, synapse=AMPA(), solver=NoSolver(wI, weights=True))
        nengo.Connection(diff, ens, synapse=NMDA(), solver=NoSolver(w1, weights=True))
        nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(w1, weights=True))
        nengo.Connection(ens, diff, synapse=NMDA(), solver=NoSolver(w2, weights=True))
        nengo.Connection(inh, diff, synapse=GABA(), solver=NoSolver(wInh, weights=True))
        nengo.Connection(ens, cleanup.input, synapse=fNMDA, solver=NoSolver(d1, weights=False))

        network.pInpt = nengo.Probe(inpt, synapse=None)
        network.pGate = nengo.Probe(gate, synapse=None)
        network.pDiff = nengo.Probe(diff.neurons, synapse=None)
        network.pEns = nengo.Probe(ens.neurons, synapse=None)
        network.pInh = nengo.Probe(inh.neurons, synapse=None)
        network.pCleanup = nengo.Probe(cleanup.output, synapse=fNMDA)

    with nengo.Simulator(network, dt=dt, progress_bar=False) as sim:
        neuron.h.init()
        sim.run(tGate+t, progress_bar=True)
        nrnReset(sim, network)

    aDiff = fNMDA.filt(sim.data[network.pDiff], dt=dt)
    aEns = fNMDA.filt(sim.data[network.pEns], dt=dt)
    xhatDiff = np.dot(aDiff, d1)
    xhatEns = np.dot(aEns, d1)

    return dict(
        times=sim.trange(),
        inpt=sim.data[network.pInpt],
        gate=sim.data[network.pGate],
        diff=xhatDiff,
        ens=xhatEns,
        cleanup=sim.data[network.pCleanup])

def run_bioneuron(nTest=1, nSeeds=1, t=20, tGate=1, dt=0.001, plot=True, load=False, thr=0.1, trainDA=0.0, testDA=0.0, seed=0):
    columns = ('seed', 'trial', 'delay length', 'error', 'error_cleanup', 'correct')
    dfs = [] 
    if load=='pkl':
        data = pd.read_pickle(f"data/delayed_response_task_bioneuron_trainDA={trainDA}_testDA={testDA}_seed{seed}.pkl")
    elif load=='npz':
        dfs = []
        for seed in range(nSeeds):
            print(f"seed {seed}")
            for n in range(nTest): 
                df = np.load(f"data/delayed_response_task_bioneuron_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
                for i in range(len(df['times'])):
                    dfs.append(pd.DataFrame([[
                        seed, n, df['times'][i]-tGate, df['error_estimate'][i], df['error_cleanup'][i], df['correct'][i]]], columns=columns))
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/delayed_response_task_bioneuron_trainDA={trainDA}_testDA={testDA}_seed{seed}.pkl")
    else:
        angles = np.linspace(0, 2*np.pi, nTest+1)[:-1]
        cues = np.array([[np.sin(angle), np.cos(angle)] for angle in angles])
        cueInpt = CueInput()
        gateInpt = GateInput(tGate)
        for n in range(nTest):
            cueInpt.set(cues[n])
            data = bioneuron_test(cueInpt, gateInpt, cues, t, dt, tGate, trainDA=trainDA, testDA=testDA, seed=seed)
            error_estimate = np.sqrt(np.square(data['inpt'][:,0]-data['ens'][:,0]) + np.square(data['inpt'][:,1]-data['ens'][:,1]))
            error_cleanup = np.sqrt(np.square(data['inpt'][:,0]-data['cleanup'][:,0]) + np.square(data['inpt'][:,1]-data['cleanup'][:,1]))
            correct = np.array([100 if error_cleanup[i]<thr else 0 for i in range(len(data['times']))])
            np.savez(f"data/delayed_response_task_bioneuron_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz",
                times=data['times'],
                inpt=data['inpt'],
                ens=data['ens'],
                cleanup=data['cleanup'],
                error_estimate=error_estimate,
                error_cleanup=error_cleanup,
                correct=correct)
            if plot:
                fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6,4)))
                ax.plot(data['times'], data['inpt'], label='target')
                ax.plot(data['times'], data['ens'], label='ens')
                ax.plot(data['times'], data['cleanup'], label='cleanup')
                ax2.plot(data['times'], correct)
                # ax.plot(data['times'], error_estimate, label='error estimate')
                # ax.plot(data['times'], error_cleanup, label='error cleanup')
                ax.set(ylabel='estimate')
                ax.legend()
                ax2.set(xlabel='time (s)', ylabel='correct')
                fig.savefig(f'plots/delayed_response_task/bioneuron_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.pdf')
                plt.close('all')

        dfs = []
        for seed in range(nSeeds):
            for n in range(nTest): 
                data = np.load(f"data/delayed_response_task_bioneuron_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
                for i in range(len(data['times'])):
                    dfs.append(pd.DataFrame([[
                        trainDA, testDA, seed, n, data['times'][i]-tGate, data['error_estimate'][i], data['error_cleanup'][i], data['correct'][i]]], columns=columns))
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/delayed_response_task_bioneuron_trainDA={trainDA}_testDA={testDA}_seed{seed}.pkl")

    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6,4)))
    sns.lineplot(data=data, x='delay length', y='error', ax=ax)
    sns.lineplot(data=data, x='delay length', y='correct', ax=ax2)
    ax.set(xlim=((0, t)), xticks=((0, t)), ylim=((0, 0.4)), yticks=((0, 1)), ylabel='Error (Euclidean)')
    ax2.set(xlim=((0, t)), xticks=((0, t)), ylim=((0, 100)), yticks=((0, 100)), ylabel='Percent Correct', xlabel='Delay Length (s)')
    plt.tight_layout()
    fig.savefig(f'plots/delayed_response_task/baseline.pdf')

run_bioneuron()