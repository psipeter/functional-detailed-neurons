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
from utils import LearningNode, trainD, fitSinusoid, plotActivities, checkTrain
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette()
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

fAMPA = DoubleExp(0.55e-3, 2.2e-3)
fNMDA = DoubleExp(10.6e-3, 285e-3)
fGABA = DoubleExp(0.5e-3, 1.5e-3)
fSmooth = DoubleExp(1e-2, 1e-1)
m = Uniform(20, 40)
iInh = Uniform(0.5, 1)
r = 1.4
nEns = 100
nPre = 1000
dt = 1e-3

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


def networkBaseline(seed, cueInpt, gateInpt, cues):
    eInh = Choice([[1,1]])
    ePre = sphere.sample(nPre, 2, rng=np.random.RandomState(seed=seed))
    eTarget = sphere.sample(nEns, 2, rng=np.random.RandomState(seed=seed))
    wInh = -1e-1*np.ones((nEns, nEns))
    network = nengo.Network(seed=seed)
    with network:    
        inpt = nengo.Node(lambda t, x: cueInpt.get(), size_in=2, size_out=2)
        gate = nengo.Node(lambda t, x: gateInpt.get(t), size_in=2, size_out=2)
        pre = nengo.Ensemble(nPre, 2, radius=r, max_rates=m, encoders=ePre, seed=seed)
        preI = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        diff = nengo.Ensemble(nEns, 2, radius=r, max_rates=m, encoders=eTarget, seed=seed)
        ens = nengo.Ensemble(nEns, 2, radius=r, max_rates=m, encoders=eTarget, seed=seed)
        inh = nengo.Ensemble(nEns, 2, radius=r, max_rates=m, intercepts=iInh, encoders=eInh, seed=seed+1)
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

def goBaseline(network, t=8, tGate=2):
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

def baseline(nCues=16, nSeeds=10, tTest=20, tGate=1, plot=True, load=False, thr=0.1):
    columns = ('seed', 'trial', 'delay length', 'error', 'error_cleanup', 'correct')
    dfs = [] 
    if load=='pkl':
        data = pd.read_pickle(f"data/DRT_baseline.pkl")
    elif load=='npz':
        dfs = []
        for seed in range(nSeeds):
            print(f"seed {seed}")
            for n in range(nCues): 
                df = np.load(f"data/DRT_baseline_seed{seed}_trial{n}.npz")
                for i in range(len(df['times'])):
                    dfs.append(pd.DataFrame([[
                        seed, n, df['times'][i]-tGate, df['error_estimate'][i], df['error_cleanup'][i], df['correct'][i]]], columns=columns))
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/DRT_baseline.pkl")
    else:
        angles = np.linspace(0, 2*np.pi, nCues+1)[:-1]
        cues = np.array([[np.sin(angle), np.cos(angle)] for angle in angles])
        cueInpt = CueInput()
        gateInpt = GateInput(tGate)
        for seed in range(nSeeds):
            network = networkBaseline(seed, cueInpt, gateInpt, cues)
            for n in range(nCues):
                cueInpt.set(cues[n])
                data = goBaseline(network, t=tTest, tGate=tGate)
                error_estimate = np.sqrt(np.square(data['inpt'][:,0]-data['ens'][:,0]) + np.square(data['inpt'][:,1]-data['ens'][:,1]))
                error_cleanup = np.sqrt(np.square(data['inpt'][:,0]-data['cleanup'][:,0]) + np.square(data['inpt'][:,1]-data['cleanup'][:,1]))
                correct = np.array([100 if error_cleanup[i]<thr else 0 for i in range(len(data['times']))])
                np.savez(f"data/DRT_baseline_seed{seed}_trial{n}.npz",
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
                    fig.savefig(f'plots/DRT/baseline_seed{seed}_trial{n}.pdf')
                    plt.close('all')
        dfs = []
        for seed in range(nSeeds):
            print(f"seed {seed}")
            for n in range(nCues): 
                data = np.load(f"data/DRT_baseline_seed{seed}_trial{n}.npz")
                for i in range(len(data['times'])):
                    dfs.append(pd.DataFrame([[
                        seed, n, data['times'][i]-tGate, data['error_estimate'][i], data['error_cleanup'][i], data['correct'][i]]], columns=columns))
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/DRT_baseline.pkl")

    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6,4)))
    sns.lineplot(data=data, x='delay length', y='error', ax=ax)
    sns.lineplot(data=data, x='delay length', y='correct', ax=ax2)
    ax.set(xlim=((0, tTest)), xticks=((0, tTest)), ylim=((0, 0.4)), yticks=((0, 1)), ylabel='Error (Euclidean)')
    ax2.set(xlim=((0, tTest)), xticks=((0, tTest)), ylim=((0, 100)), yticks=((0, 100)), ylabel='Percent Correct', xlabel='Delay Length (s)')
    plt.tight_layout()
    fig.savefig(f'plots/DRT/baseline.pdf')


def makeSignalCircle(t, rad=1, rms=0.2, seed=0):
    phase = np.random.RandomState(seed=seed).uniform(0, 1)
    stim = nengo.processes.WhiteSignal(period=t, high=1, rms=rms, seed=seed)
    stim2 = nengo.processes.WhiteSignal(period=t, high=1, rms=rms, seed=50+seed)
    with nengo.Network() as model:
        inpt = nengo.Node(stim)
        inpt2 = nengo.Node(stim2)
        probe = nengo.Probe(inpt, synapse=None)
        probe2 = nengo.Probe(inpt2, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    stim_func = lambda tt: rad*np.sin(2*np.pi*(tt/t+phase)) + sim.data[probe][int(tt/dt)]
    stim_func2 = lambda tt: rad*np.cos(2*np.pi*(tt/t+phase)) + sim.data[probe2][int(tt/dt)]
    return stim_func, stim_func2

def goTrain(trainDA, t=10, seed=0,
    stim_func1=lambda t: 0, stim_func2=lambda t: 0, stim_func3=lambda t: [0, 0] if t<3 else [1,1], 
    dB=None, eB=None, wB=None, d0=None, e0=None, w0=None, d1=None, e1=None, w1=None, dI=None, eI=None, wI=None,
    learn0=False, check0=False, learn1=False, check1=False, learn2=False, check2=False):

    ePre = sphere.sample(nPre, 2, rng=np.random.RandomState(seed=seed))
    eTarget = sphere.sample(nEns, 2, rng=np.random.RandomState(seed=seed))
    with nengo.Network() as model:
        inpt1 = nengo.Node(stim_func1)
        inpt2 = nengo.Node(stim_func2)
        gate = nengo.Node(stim_func3)
        inpt = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        const = nengo.Node([1,1])
        pre = nengo.Ensemble(nPre, 2, radius=r, max_rates=m, encoders=ePre, seed=seed)
        bias = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
        nengo.Connection(inpt1, inpt[0], synapse=None)
        nengo.Connection(inpt2, inpt[1], synapse=None)
        nengo.Connection(inpt, pre, synapse=None)

        if learn0:
            tarEns = nengo.Ensemble(nEns, 2, radius=r, max_rates=m, encoders=eTarget, neuron_type=ReLu(), seed=seed)
            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)  
            nengo.Connection(inpt, tarEns, synapse=fAMPA, seed=seed)
            nengo.Connection(bias, tarEns, synapse=fAMPA, seed=seed)
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

        if check0:
            tarEns = nengo.Ensemble(nEns, 2, radius=r, max_rates=m, encoders=eTarget, neuron_type=ReLu(), seed=seed)  # feedforward AMPA
            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)  
            nengo.Connection(inpt, tarEns, synapse=fAMPA, seed=seed)
            nengo.Connection(bias, tarEns, synapse=fAMPA, seed=seed)
            nengo.Connection(bias, ens, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(pre, ens, synapse=AMPA(), solver=NoSolver(w0, weights=True))

        if learn1:
            pre2 = nengo.Ensemble(nPre, 2, radius=r, max_rates=m, encoders=ePre, seed=seed)
            ens2 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
            ens3 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
            nengo.Connection(bias, ens, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens2, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens3, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(inpt, pre2, synapse=fNMDA)
            nengo.Connection(pre, ens, synapse=AMPA(), solver=NoSolver(w0, weights=True))
            nengo.Connection(pre2, ens3, synapse=AMPA(), solver=NoSolver(w0, weights=True))
            conn1 = nengo.Connection(ens, ens2, synapse=NMDA(), solver=NoSolver(np.zeros((nEns, nEns)), weights=True))
            node1 = LearningNode(ens, ens2, 2, conn=conn1, d=d1, e=e1, w=w1, eRate=3e-9, dRate=0)
            nengo.Connection(ens.neurons, node1[:nEns], synapse=fNMDA)
            nengo.Connection(ens2.neurons, node1[nEns: 2*nEns], synapse=fSmooth)
            nengo.Connection(ens3.neurons, node1[2*nEns: 3*nEns], synapse=fSmooth)
            nengo.Connection(node1, ens2.neurons, synapse=None)
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)

        if check1:
            pre2 = nengo.Ensemble(nPre, 2, radius=r, max_rates=m, encoders=ePre, seed=seed)
            ens2 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
            ens3 = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=trainDA), seed=seed)
            nengo.Connection(bias, ens, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens2, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(bias, ens3, synapse=AMPA(), solver=NoSolver(wB, weights=True))
            nengo.Connection(inpt, pre2, synapse=fNMDA)
            nengo.Connection(pre, ens, synapse=AMPA(), solver=NoSolver(w0, weights=True))
            nengo.Connection(pre2, ens3, synapse=AMPA(), solver=NoSolver(w0, weights=True))
            nengo.Connection(ens, ens2, synapse=NMDA(), solver=NoSolver(w1, weights=True))
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)

        if learn2:
            preI = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
            inh = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Interneuron', DA=trainDA), seed=seed+1)
            tarInh = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=iInh, radius=r, encoders=Choice([[1,1]]), neuron_type=ReLu(), seed=seed+1)
            nengo.Connection(gate, preI, synapse=None)
            nengo.Connection(gate, tarInh, synapse=fAMPA, seed=seed)
            connI = nengo.Connection(preI, inh, synapse=AMPA(), solver=NoSolver(np.zeros((nPre, nEns)), weights=True))
            nodeI = LearningNode(preI, inh, 2, conn=connI, d=dI, e=eI, w=wI, eRate=5e-7, dRate=1e-6)
            nengo.Connection(preI.neurons, nodeI[:nPre], synapse=fAMPA)
            nengo.Connection(inh.neurons, nodeI[nPre: nPre+nEns], synapse=fSmooth)
            nengo.Connection(tarInh.neurons, nodeI[nPre+nEns: nPre+nEns+nEns], synapse=fSmooth)
            nengo.Connection(gate, nodeI[nPre+nEns+nEns:], synapse=None)
            nengo.Connection(nodeI, inh.neurons, synapse=None)
            pInh = nengo.Probe(inh.neurons, synapse=None)
            pTarInh = nengo.Probe(tarInh.neurons, synapse=None)  

        if check2:
            preI = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
            inh = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Interneuron', DA=trainDA), seed=seed+1)
            tarInh = nengo.Ensemble(nEns, 2, max_rates=m, intercepts=iInh, radius=r, encoders=Choice([[1,1]]), neuron_type=ReLu(), seed=seed+1)
            nengo.Connection(gate, tarInh, synapse=fAMPA, seed=seed)
            nengo.Connection(preI, inh, synapse=AMPA(), solver=NoSolver(wI, weights=True))
            nengo.Connection(gate, preI, synapse=None)
            pInh = nengo.Probe(inh.neurons, synapse=None)
            pTarInh = nengo.Probe(tarInh.neurons, synapse=None)

        pInpt = nengo.Probe(inpt, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        neuron.h.init()
        sim.run(t, progress_bar=True)
        nrnReset(sim, model)

    if learn0:
        dB, eB, wB = nodeB.d, nodeB.e, nodeB.w
        d0, e0, w0 = node0.d, node0.e, node0.w
    if learn1:
        e1, w1 = node1.e, node1.w
    if learn2:
        dI, eI, wI = nodeI.d, nodeI.e, nodeI.w

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        ens=sim.data[pEns],
        tarEns=sim.data[pTarEns] if learn0 or check0 else None,
        ens2=sim.data[pEns2] if learn1 or check1 else None,
        ens3=sim.data[pEns3] if learn1 or check1 else None,
        inh=sim.data[pInh] if learn2 or check2 else None,
        tarInh=sim.data[pTarInh] if learn2 or check2 else None,
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
    )


def train(trainDA, seed, load=[], nTrain=20, tTrain=10):

    if 0 in load:
        data = np.load(f"data/DRT_trainDA{trainDA}_seed{seed}.npz")
        d0, e0, w0 = data['d0'], data['e0'], data['w0']
        dB, eB, wB = data['dB'], data['eB'], data['wB']
    else:
        print('train d0, e0, w0 from pre to diff')
        d0, e0, w0 = None, None, None
        dB, eB, wB = None, None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, seed=n)
            data = goTrain(trainDA=trainDA, learn0=True, t=tTrain, seed=seed, stim_func1=stim_func1, stim_func2=stim_func2, 
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0)
            dB, eB, wB = data['dB'], data['eB'], data['wB']
            d0, e0, w0 = data['d0'], data['e0'], data['w0']
            np.savez(f"data/DRT_trainDA{trainDA}_seed{seed}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0)
        # stim_func1, stim_func2 = makeSignalCircle(tTrain, seed=0)
        # data = goTrain(trainDA=trainDA, check0=True, t=tTrain, seed=seed, stim_func1=stim_func1, stim_func2=stim_func2,
        #     wB=wB,
        #     w0=w0)
        # aEns = fSmooth.filt(data['ens'], dt=dt)
        # aTar = fSmooth.filt(data['tarEns'], dt=dt)
        # checkTrain(data['times'], aEns, aTar, stage=0)

    if 1 in load:
        data = np.load(f"data/DRT_trainDA{trainDA}_seed{seed}.npz")
        d1 = data['d1']
    else:
        print('train d1 for diff/ens to compute identity')
        targets = np.zeros((nTrain, int(tTrain/dt), 2))
        spikes = np.zeros((nTrain, int(tTrain/dt), nEns))
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, seed=n)
            data = goTrain(trainDA=trainDA, check0=True, t=tTrain, seed=seed, stim_func1=stim_func1, stim_func2=stim_func2,
                wB=wB,
                w0=w0)
            targets[n] = fNMDA.filt(fAMPA.filt(data['inpt'], dt=dt), dt=dt)
            spikes[n] = data['ens']
        d1 = trainD(spikes, targets, nTrain, fNMDA, dt=dt)
        np.savez(f"data/DRT_trainDA{trainDA}_seed{seed}.npz",
            dB=dB, eB=eB, wB=wB,
            d0=d0, e0=e0, w0=w0,
            d1=d1)
        # times = data['times']
        # inpt = data['inpt']
        # target = fNMDA.filt(fAMPA.filt(data['inpt'], dt=dt), dt=dt)
        # aEns = fNMDA.filt(data['ens'], dt=dt)
        # xhat = np.dot(aEns, d1)
        # fig, ax = plt.subplots()
        # ax.plot(times, target, label='target')
        # ax.plot(times, xhat, label='xhat')
        # ax.legend()
        # ax.set(yticks=((-1,1)))
        # fig.savefig(f'plots/DRT/decode_trainDA{trainDA}_seed{seed}.pdf')

    if 2 in load:
        data = np.load(f"data/DRT_trainDA{trainDA}_seed{seed}.npz")
        e1, w1 = data['e1'], data['w1']
    else:
        print('train e1, w1 from diff/ens to ens')
        e1, w1 = None, None
        for n in range(nTrain):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, seed=n)
            data = goTrain(trainDA=trainDA, learn1=True, t=tTrain, seed=seed, stim_func1=stim_func1, stim_func2=stim_func2,
                wB=wB,
                w0=w0,
                d1=d1, e1=e1, w1=w1)
            e1, w1 = data['e1'], data['w1']
            np.savez(f"data/DRT_trainDA{trainDA}_seed{seed}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                d1=d1,
                e1=e1, w1=w1)
        # stim_func1, stim_func2 = makeSignalCircle(tTrain, seed=0)
        # data = goTrain(trainDA=trainDA, check1=True, t=tTrain, seed=seed, stim_func1=stim_func1, stim_func2=stim_func2,
        #     wB=wB,
        #     w0=w0,
        #     w1=w1)
        # aEns = fSmooth.filt(data['ens2'], dt=dt)
        # aTar = fSmooth.filt(data['ens3'], dt=dt)
        # checkTrain(data['times'], aEns, aTar, stage=2)

    if 3 in load:
        data = np.load(f"data/DRT_trainDA{trainDA}_seed{seed}.npz")
        dI, eI, wI = data['dI'], data['eI'], data['wI']
    else:
        print('train dI, eI, wI from preI to inh')
        dI, eI, wI = None, None, None
        for n in range(2):
            stim_func1, stim_func2 = makeSignalCircle(tTrain, seed=n)
            data = goTrain(trainDA=trainDA, learn2=True, t=tTrain, seed=seed, stim_func1=stim_func1, stim_func2=stim_func2,
                dI=dI, eI=eI, wI=wI)
            dI, eI, wI = data['dI'], data['eI'], data['wI']
            np.savez(f"data/DRT_trainDA{trainDA}_seed{seed}.npz",
                dB=dB, eB=eB, wB=wB,
                d0=d0, e0=e0, w0=w0,
                d1=d1,
                e1=e1, w1=w1,
                dI=dI, eI=eI, wI=wI)
        # stim_func1, stim_func2 = makeSignalCircle(tTrain, seed=0)
        # data = goTrain(trainDA=trainDA, check2=True, t=tTrain, seed=seed, stim_func1=stim_func1, stim_func2=stim_func2,
        #     wI=wI)
        # aEns = fSmooth.filt(data['inh'], dt=dt)
        # aTar = fSmooth.filt(data['tarInh'], dt=dt)
        # checkTrain(data['times'], aEns, aTar, stage=3)


def goTest(cueInpt, gateInpt, cues, t, tGate, trainDA=0.0, testDA=0.0, seed=0):

    weights = np.load(f"data/DRT_trainDA{trainDA}_seed{seed}.npz")
    wB, wI, w0, w1, d1 = weights['wB'], weights['wI'], weights['w0'], weights['w1'], weights['d1'],
    ePre = sphere.sample(nPre, 2, rng=np.random.RandomState(seed=seed))
    wInh = -1e-1*np.ones((nEns, nEns))
    with nengo.Network() as network:
        inpt = nengo.Node(lambda t, x: cueInpt.get(), size_in=2, size_out=2)
        gate = nengo.Node(lambda t, x: gateInpt.get(t), size_in=2, size_out=2)
        bias = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        pre = nengo.Ensemble(nPre, 2, radius=r, max_rates=m, encoders=ePre, seed=seed)
        preI = nengo.Ensemble(nPre, 2, max_rates=m, seed=seed)
        diff = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=testDA), seed=seed)
        ens = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Pyramidal', DA=testDA), seed=seed)
        inh = nengo.Ensemble(nEns, 2, neuron_type=NEURON('Interneuron', DA=testDA), seed=seed+1)
        cleanup = nengo.networks.AssociativeMemory(cues, n_neurons=nEns, seed=seed)
        cleanup.add_wta_network(inhibit_synapse=fGABA, inhibit_scale=1.3)
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
        nengo.Connection(ens, diff, synapse=NMDA(), solver=NoSolver(-w1, weights=True))
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

def test(nCues=1, nSeeds=1, tTest=20, tGate=1, plot=True, load=False, thr=0.1, trainDA=0.0, testDA=0.0, seed=0):
    columns = ('seed', 'trial', 'delay length', 'error', 'error_cleanup', 'correct')
    dfs = [] 
    if load=='pkl':
        data = pd.read_pickle(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}.pkl")
    elif load=='npz':
        dfs = []
        # for seed in range(nSeeds):
            # print(f"seed {seed}")
        for n in range(nCues): 
            print(f"cue {n}")
            df = np.load(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
            for i in range(len(df['times'])):
                dfs.append(pd.DataFrame([[
                    seed, n, df['times'][i]-tGate, df['error_estimate'][i], df['error_cleanup'][i], df['correct'][i]]], columns=columns))
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}.pkl")
    else:
        angles = np.linspace(0, 2*np.pi, nCues+1)[:-1]
        cues = np.array([[np.sin(angle), np.cos(angle)] for angle in angles])
        cueInpt = CueInput()
        gateInpt = GateInput(tGate)
        for n in range(nCues):
            if n<=14: continue
            print(f"Testing cue {n} at ({cues[n][0]:.2}, {cues[n][1]:.2})")
            cueInpt.set(cues[n])
            data = goTest(cueInpt, gateInpt, cues, tTest, tGate, trainDA=trainDA, testDA=testDA, seed=seed)
            error_estimate = np.sqrt(np.square(data['inpt'][:,0]-data['ens'][:,0]) + np.square(data['inpt'][:,1]-data['ens'][:,1]))
            error_cleanup = np.sqrt(np.square(data['inpt'][:,0]-data['cleanup'][:,0]) + np.square(data['inpt'][:,1]-data['cleanup'][:,1]))
            correct = np.array([100 if error_cleanup[i]<thr else 0 for i in range(len(data['times']))])
            np.savez(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz",
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
                ax.set(ylabel='estimate')
                ax.legend()
                ax2.set(xlabel='time (s)', ylabel='correct')
                fig.savefig(f'plots/DRT/trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.pdf')
                plt.close('all')

        dfs = []
        for seed in range(nSeeds):
            for n in range(nCues): 
                data = np.load(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
                for i in range(len(data['times'])):
                    dfs.append(pd.DataFrame([[
                        seed, n, data['times'][i]-tGate, data['error_estimate'][i], data['error_cleanup'][i], data['correct'][i]]], columns=columns))
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}.pkl")

    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6,4)))
    sns.lineplot(data=data, x='delay length', y='error', ax=ax)
    sns.lineplot(data=data, x='delay length', y='correct', ax=ax2)
    ax.set(xlim=((0, tTest)), xticks=((0, tTest)), ylim=((0, 0.4)), yticks=((0, 1)), ylabel='Error (Euclidean)')
    ax2.set(xlim=((0, tTest)), xticks=((0, tTest)), ylim=((0, 100)), yticks=((0, 100)), ylabel='Percent Correct', xlabel='Delay Length (s)')
    plt.tight_layout()
    fig.savefig(f'plots/DRT/trainDA={trainDA}_testDA{testDA}_seed{seed}.pdf')

def load_and_plot(nCues=1, seeds=[], tTest=20, tGate=1, load=False, trainDA=0.0, testDA=0.0):
    columns = ('seed', 'trial', 'delay length', 'error', 'error_cleanup', 'correct')
    dfs = [] 
    if load=='pkl':
        data = pd.read_pickle(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}.pkl")
    elif load=='npz':
        dfs = []
        for seed in seeds:
            print(f"load data from seed {seed}")
            for n in range(nCues): 
                print(f"cue {n}")
                # df = np.load(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
                df = np.load(f"data/ncues_8/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
                for i in range(len(df['times'])):
                    dfs.append(pd.DataFrame([[
                        seed, n, df['times'][i]-tGate, df['error_estimate'][i], df['error_cleanup'][i], df['correct'][i]]], columns=columns))
        print('concatenate and save')
        data = pd.concat(dfs, ignore_index=True)
        # data.to_pickle(f"data/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
        data.to_pickle(f"data/ncues_8/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
    print('plot')
    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6,4)))
    sns.lineplot(data=data, x='delay length', y='error', ax=ax)
    sns.lineplot(data=data, x='delay length', y='correct', ax=ax2)
    ax.set(xlim=((0, tTest)), xticks=((0, tTest)), ylim=((0, 0.4)), yticks=((0, 1)), ylabel='Error (Euclidean)')
    ax2.set(xlim=((0, tTest)), xticks=((0, tTest)), ylim=((0, 100)), yticks=((0, 100)), ylabel='Percent Correct', xlabel='Delay Length (s)')
    plt.tight_layout()
    # fig.savefig(f'plots/DRT/trainDA={trainDA}_testDA{testDA}_ncues8.pdf')
    fig.savefig(f'plots/DRT/trainDA={trainDA}_testDA{testDA}_nCues{nCues}_nSeeds{len(seeds)}.pdf')


def plot_euclidean_forgetting(nCues=8, seeds=[], tTest=20, tGate=1, load='npz', trainDA=0.0, testDA=0.0):
    columns = ('seed', 'trial', 'delay_length', 'error', 'error_cleanup', 'correct')
    dfs = [] 
    if load=='pkl':
        data = pd.read_pickle(f"data/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
    elif load=='npz':
        dfs = []
        for seed in seeds:
            print(f"load data from seed {seed}")
            for n in range(nCues): 
                print(f"cue {n}")
                # df = np.load(f"data/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
                # df = np.load(f"data/ncues_8/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
                df = np.load(f"data/20s/DRT_trainDA={trainDA}_testDA={testDA}_seed{seed}_trial{n}.npz")
                for i in range(len(df['times'])):
                    if df['times'][i] >= tGate:
                        dfs.append(pd.DataFrame([[
                            seed, n, df['times'][i]-tGate, df['error_estimate'][i], df['error_cleanup'][i], df['correct'][i]]], columns=columns))
        print('concatenate and save')
        data = pd.concat(dfs, ignore_index=True)
        data.to_pickle(f"data/20s/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
        # data.to_pickle(f"data/ncues_8/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
    # print('plot')
    # fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=((6,4)))
    # sns.lineplot(data=data, x='delay_length', y='error', ax=ax)
    # sns.lineplot(data=data, x='delay_length', y='correct', ax=ax2)
    # ax.set(xlim=((0, tTest+tGate)), xticks=((0, tGate, tTest+tGate)), ylim=((0, 0.4)), yticks=((0, 1)), ylabel='Error (Euclidean)')
    # ax2.set(xlim=((0, tTest+tGate)), xticks=((0, tGate, tTest+tGate)), ylim=((0, 100)), yticks=((0, 100)), ylabel='Percent Correct', xlabel='Delay Length (s)')
    # plt.tight_layout()
    # fig.savefig(f'plots/DRT/trainDA={trainDA}_testDA{testDA}_test.pdf')
    # fig.savefig(f'plots/DRT/trainDA={trainDA}_testDA{testDA}_nCues{nCues}_nSeeds{len(seeds)}.pdf')

def exponential(t, baseline, tau):
    return baseline * np.exp(-t/tau)

# def fit_forgetting_individual(seeds=[], tTest=20, tStart=0, tStep=1, trainDA=0.0, testDA=0.0):
#     columns = ('seed', 'delay_length', 'correct', 'scaled')
#     data = pd.read_pickle(f"data/20s/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
#     tSamples = np.arange(tStart, tTest+tStep, tStep)
#     fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((5.2,2)), sharey=True)
#     # fig2, ax = plt.subplots(figsize=((6,3)))
#     baselines = []
#     taus = []
#     half_lives = []
#     for seed in seeds:
#         corrects = []
#         for t in range(len(tSamples)-1):
#             times = data.query("seed==@seed")['delay_length'].to_numpy()
#             t0 = tSamples[t]
#             t1 = tSamples[t+1]
#             mean_correct_over_window = np.mean(data.query("seed==@seed & delay_length>=@t0 & delay_length<@t1")['correct'].to_numpy())
#             corrects.append(mean_correct_over_window)
#         corrects = np.array(corrects)
#         idx_start = np.argmax(corrects)
#         t_monotonic = tSamples[1:][idx_start:]
#         c_monotonic = corrects[idx_start:]
#         t_left_aligned = t_monotonic - t_monotonic[0]
#         # params, covariance = sp.optimize.curve_fit(exponential, t_monotonic, c_monotonic)
#         # params, covariance = sp.optimize.curve_fit(exponential, t_left_aligned, c_monotonic)
#         baseline = c_monotonic[0]
#         def left_aligned_exponential(t, tau):
#             return baseline * np.exp(-t/tau)
#         params, covariance = sp.optimize.curve_fit(left_aligned_exponential, t_left_aligned, c_monotonic)
#         # print(f"seed = {seed}")
#         # print(f"baseline performance = {np.around(baseline, 1)}")
#         # print(f"performance half life = {np.around(params[0]*np.log(2), 1)}")
#         baselines.append(baseline)
#         taus.append(params[0])
#         half_lives.append(params[0]*np.log(2))
#         ax.plot(tSamples[1:], corrects, label=seed)
#         ax2.scatter(t_monotonic[0], left_aligned_exponential(t_left_aligned, params[0])[0], label=seed)
#         ax2.plot(t_monotonic, left_aligned_exponential(t_left_aligned, params[0]), label=seed)
#     ax.set(xlim=((0, tTest+tStep)), xticks=((0, tTest)), yticks=((0, 100)), ylim=((-10, 110)),
#         ylabel=f'% Correct', xlabel='Delay Length (s)', title="raw simulated data")
#     ax2.set(xlim=((0, tTest+tStep)), xticks=((0, tTest)), xlabel="Time (s)", title="best fit exponential")
#     fig.tight_layout()
#     # fig.legend(loc='upper right')
#     # fig.savefig(f'plots/figures/forgetting_curves.svg')

#     sns.set(context='paper', style='whitegrid', font='CMU Serif',
#         rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=((5.2, 0.5)))
#     axes[0].scatter(np.array(baselines), np.zeros_like(baselines), s=20, color=palette[0])
#     axes[0].scatter(np.median(baselines), 0, s=150, facecolors="none", edgecolors=palette[0])
#     axes[0].set(xlim=((50, 100)), xticks=((50, 60, 70, 80, 90, 100)), yticks=(()))
#     axes[1].scatter(np.array(half_lives), np.zeros_like(half_lives), s=20, color=palette[0])
#     axes[1].scatter(np.median(half_lives), 0, s=150, facecolors="none", edgecolors=palette[0])
#     axes[1].set(yticks=(()))
#     axes[1].set_xscale('log')
#     axes[1].set_xticks([1, 10, 30, 60, 120, 300])
#     axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#     # fig.savefig('plots/figures/model_forgetting_parameters.svg')

#     fig, ax = plt.subplots(figsize=((5.2, 0.5)))
#     ax.set(ylabel='biologically detailed\ncognitive network')
#     fig.savefig("plots/figures/label.svg")

#     # fig2.savefig(f"plots/DRT/forgetting_curve_baselines.svg")

#     # fig3, ax3 = plt.subplots(figsize=((3, 1)))
#     # x = np.array(half_lives)
#     # y = np.zeros_like(x)
#     # ax3.axhline(0, color='k', alpha=0.2)
#     # ax3.scatter(x, y, s=100, color=palette[1])
#     # ax3.scatter(np.median(half_lives), 0, s=150, facecolors="none", edgecolors=palette[1])
#     # ax3.set(xlim=((1, 10)), xticks=((1, 10)))
#     # fig3.savefig(f"plots/DRT/forgetting_curve_halflives1.svg")

#     # fig4, ax4 = plt.subplots(figsize=((2, 1)))
#     # x = np.array(half_lives)
#     # y = np.zeros_like(x)
#     # ax4.axhline(0, color='k', alpha=0.2)
#     # ax4.scatter(x, y, s=100, color=palette[1])
#     # ax4.scatter(np.median(half_lives), 0, s=150, facecolors="none", edgecolors=palette[1])
#     # ax4.set(xlim=((10, 30)), xticks=((10, 30)))
#     # fig4.savefig(f"plots/DRT/forgetting_curve_halflives2.svg")

#     print("baseline correct: \t", "min", np.around(np.min(baselines), 2), "max", np.around(np.max(baselines), 2), "mean", np.around(np.mean(baselines), 2), "median", np.around(np.median(baselines), 2))
#     print("performance half life \t", "min", np.around(np.min(half_lives), 2), "max", np.around(np.max(half_lives), 2), "mean", np.around(np.mean(half_lives), 2), "median", np.around(np.median(half_lives), 2))

# def fit_forgetting_group(seeds=[], tTest=20, tStart=0, tStep=1, trainDA=0.0, testDA=0.0, median_baseline=0, median_tau=0):
#     columns = ('seed', 'delay_length', 'correct')
#     data = pd.read_pickle(f"data/20s/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
#     tSamples = np.arange(tStart, tTest+tStep, tStep)
#     dfs = []
#     for seed in seeds:
#         corrects = []
#         for t in range(len(tSamples)-1):
#             times = data.query("seed==@seed")['delay_length'].to_numpy()
#             t0 = tSamples[t]
#             t1 = tSamples[t+1]
#             mean_correct_over_window = np.mean(data.query("seed==@seed & delay_length>=@t0 & delay_length<@t1")['correct'].to_numpy())
#             corrects.append(mean_correct_over_window)
#         corrects = np.array(corrects)
#         idx_start = np.argmax(corrects)
#         t_monotonic = tSamples[1:][idx_start:]
#         c_monotonic = corrects[idx_start:]
#         t_left_aligned = t_monotonic - t_monotonic[0]
#         for i in range(len(t_monotonic)):
#             dfs.append(pd.DataFrame([[str(seed), t_monotonic[i], c_monotonic[i]]], columns=columns))
#     df = pd.concat(dfs, ignore_index=True)
#     # all_times = df['delay_length'].to_numpy().ravel()
#     # all_corrects = df['correct'].to_numpy().ravel()
#     # params, covariance = sp.optimize.curve_fit(exponential, all_times, all_corrects)
#     # stds = np.sqrt(np.diag(covariance))
#     # baseline_performance = params[0]
#     # half_life = params[1] * np.log(2)
#     # print(f"baseline performance = {np.around(params[0], 1)} +/- {np.around(stds[0],1)}")
#     # print(f"performance tau = {np.around(params[1],1)} +/- {np.around(stds[1],1)}s")
#     # print(f"performance half life = {np.around(params[1]*np.log(2),1)} +/- {np.around(stds[1]*np.log(2),1)}s")

#     fig, ax = plt.subplots(figsize=((6,3)))
#     sns.lineplot(data=df, x='delay_length', y='correct', label="simulated data")
#     # ax.plot(tSamples, exponential(tSamples, params[0], params[1]), label="best fit exponential")
#     ax.plot(tSamples, exponential(tSamples, median_baseline, median_tau), label="best fit exponential")
#     sns.lineplot(data=df, x='delay_length', y='correct', hue="seed", alpha=0.2)
#     ax.set(xlim=((0, tTest)), xticks=((0, tTest)), yticks=((0, 100)), ylim=((-10, 110)), ylabel=f'% Correct', xlabel='Delay Length (s)')
#     fig.tight_layout()
#     fig.legend(loc='upper right')
#     fig.savefig(f'plots/DRT/forgetting_curve_group.pdf')
#     fig.savefig(f'plots/DRT/forgetting_curve_group.svg')


def fit_forgetting(seeds=[], tTest=20, tStart=0.5, trainDA=0.0, testDA=0.0):
    columns = ('seed', 'delay_length', 'correct', 'scaled')
    # data = pd.read_pickle(f"data/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
    # data = pd.read_pickle(f"data/ncues_8/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
    data = pd.read_pickle(f"data/20s/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
    tSamples = np.arange(tStart, tTest+0.5, 0.5)
    nTrials = np.max(data['trial'].unique())+1
    dfs = []
    corrects = []
    for seed in seeds:
        for t in tSamples:
            percent_correct_t = np.mean(data.query("seed==@seed & delay_length==@t")['correct'].to_numpy())
            scaled_correct_t = (percent_correct_t - (1/8)) / (7/8)
            dfs.append(pd.DataFrame([[seed, t, percent_correct_t, scaled_correct_t]], columns=columns))
    df = pd.concat(dfs, ignore_index=True)
    corrects = np.array([df.query("delay_length==@t")['correct'].to_numpy() for t in tSamples])
    scaled = np.array([df.query("delay_length==@t")['scaled'].to_numpy() for t in tSamples])
    times = np.array([tSamples for s in seeds]).T
    # print(times.ravel())
    # print(corrects.ravel())
    params, covariance = sp.optimize.curve_fit(exponential, times.ravel(), corrects.ravel())
    stds = np.sqrt(np.diag(covariance))
    baseline_performance = params[0]
    half_life = params[1] * np.log(2)
    b_string = f"B = {np.around(params[0],1)} +/- {np.around(stds[0],1)}"
    tau_string = f"tau = {np.around(params[1],1)} +/- {np.around(stds[1],1)}s"
    print(b_string)
    print(tau_string)
    print(f"baseline performance = {np.around(baseline_performance, 1)}")
    print(f"performance half life = {np.around(half_life, 1)}")

    # params2, covariance2 = sp.optimize.curve_fit(exponential, times.ravel(), scaled.ravel())
    # stds2 = np.sqrt(np.diag(covariance2))
    # baseline_performance = (7/8)*params2[0] + (1/8)
    # half_life = params2[1] * np.log(2)
    # b_string = f"B = {np.around(params2[0],1)} +/- {np.around(stds2[0],1)}"
    # tau_string = f"tau = {np.around(params2[1],1)} +/- {np.around(stds2[1],1)}s"
    # print(b_string)
    # print(tau_string)
    # print(f"baseline performance = {baseline_performance}")
    # print(f"performance half life = {half_life}")

    fig, ax = plt.subplots(figsize=((6,3)))
    sns.lineplot(data=df, x='delay_length', y='correct', label='simulated data')
    # sns.lineplot(data=df, x='delay_length', y='correct', hue='seed')
    # ax.plot(tSamples, exponential(tSamples, params[0], params[1]), label=fr"$y(t) = {np.around(params[0], 1)}$ exp$(-t~/~{np.around(params[1], 1)})$")
    ax.plot(tSamples, exponential(tSamples, params[0], params[1]), label="best fit exponential")
    # ax.plot(tSamples, exponential(tSamples, params2[0], params2[1]), label="best fit exponential 2")
    # ax.scatter(tau_half, (params[0]-12.5)/2, color='k', label="performance half-life")
    # ax.plot(tSamples, exponential(tSamples, 92, 10), label="best fit (empirical, lower bound)")
    # ax.plot(tSamples, exponential(tSamples, 92, 27), label="best fit (empirical, median)")
    ax.set(xlim=((0, tTest+tStart)), xticks=((0, tTest)), yticks=((0, 100)), ylim=((0, 100)),
        ylabel=f'% Correct', xlabel='Delay Length (s)')
    plt.tight_layout()
    plt.legend(loc='upper right')
    # fig.savefig(f'plots/DRT/ncues_8/forgetting_curve.pdf')
    fig.savefig(f'plots/DRT/forgetting_curve.pdf')

def plot_baseline_halflife(seeds=[], tTest=20, tStart=0, tStep=1, trainDA=0.0, testDA=0.0):
    columns = ('seed', 'delay_length', 'correct', 'scaled')
    data = pd.read_pickle(f"data/20s/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
    tSamples = np.arange(tStart, tTest+tStep, tStep)
    model_baselines = []
    model_halflives = []
    for seed in seeds:
        corrects = []
        for t in range(len(tSamples)-1):
            times = data.query("seed==@seed")['delay_length'].to_numpy()
            t0 = tSamples[t]
            t1 = tSamples[t+1]
            mean_correct_over_window = np.mean(data.query("seed==@seed & delay_length>=@t0 & delay_length<@t1")['correct'].to_numpy())
            corrects.append(mean_correct_over_window)
        corrects = np.array(corrects)
        idx_start = np.argmax(corrects)
        t_monotonic = tSamples[1:][idx_start:]
        c_monotonic = corrects[idx_start:]
        t_left_aligned = t_monotonic - t_monotonic[0]
        baseline = c_monotonic[0]
        def left_aligned_exponential(t, tau):
            return baseline * np.exp(-t/tau)
        params, covariance = sp.optimize.curve_fit(left_aligned_exponential, t_left_aligned, c_monotonic)
        model_baselines.append(baseline)
        model_halflives.append(params[0]*np.log(2))

    pigeon_baselines = [73,77,82,87,88,89,90,91,94,94,95,96,100]
    pigeon_baselines_median = 91
    chimp_baselines = [68,70,78,89,93,93,94,98]
    chimp_baselines_median = 91 
    capuchin_baselines = [84,86,92,93,96]
    capuchin_baselines_median = 92
    rhesus_baselines = [76,77,88,93,95,95,99,99,100]
    rhesus_baselines_median = 95
    dolphin_baselines = [81,93,94,100]
    dolphin_baselines_median = 97
    rat_baselines = [93,94,99,100]
    rat_baselines_median = 99

    pigeon_halflives = [2,3,3,4,4,5,11,12,16,27,28,34,38,41,51]
    pigeon_halflives_median = 14
    chimp_halflives = [6,6,19,61,159]
    chimp_halflives_median = 19
    rhesus_halflives = [2,15,17,27,29,34,48,155,157]
    rhesus_halflives_median = 32
    rat_halflives = [2,4,23,32,34,45,55,72]
    rat_halflives_median = 35
    capuchin_halflives = [10,15,16,39,107,302,453]
    capuchin_halflives_median = 39
    dolphin_halflives = [41,52,58,62,88,222]
    dolphin_halflives_median = 60

    sns.set(context='paper', style='whitegrid', font='CMU Serif',
        rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=((5.2, 1.5)), sharey=False)

    axes[0].scatter(np.array(pigeon_baselines), 1*np.ones_like(pigeon_baselines), s=2, color=palette[0])
    axes[0].scatter(pigeon_baselines_median, 1, s=20, facecolor='none', edgecolor=palette[0])
    axes[0].scatter(np.array(chimp_baselines), 2*np.ones_like(chimp_baselines), s=2, color=palette[0])
    axes[0].scatter(chimp_baselines_median, 2, s=20, facecolor='none', edgecolor=palette[0])
    axes[0].scatter(np.array(capuchin_baselines), 3*np.ones_like(capuchin_baselines), s=2, color=palette[0])
    axes[0].scatter(capuchin_baselines_median, 3, s=20, facecolor='none', edgecolor=palette[0])
    axes[0].scatter(np.array(model_baselines), 4*np.ones_like(model_baselines), s=2, color=palette[1])
    axes[0].scatter(np.median(model_baselines), 4, s=20, facecolor='none', edgecolor=palette[1])
    axes[0].scatter(np.array(dolphin_baselines), 6*np.ones_like(dolphin_baselines), s=2, color=palette[0])
    axes[0].scatter(np.array(rhesus_baselines), 5*np.ones_like(rhesus_baselines), s=2, color=palette[0])
    axes[0].scatter(rhesus_baselines_median, 5, s=20, facecolor='none', edgecolor=palette[0])
    axes[0].scatter(dolphin_baselines_median, 6, s=20, facecolor='none', edgecolor=palette[0])
    axes[0].scatter(np.array(rat_baselines), 7*np.ones_like(rat_baselines), s=2, color=palette[0])
    axes[0].scatter(rat_baselines_median, 7, s=20, facecolor='none', edgecolor=palette[0])
    axes[0].set(xlim=((50, 101)), xticks=((50, 60, 70, 80, 90, 100)),
        xlabel="zero-delay performance (% correct)",
        ylim=((0.5, 7.5)), yticks=((1,2,3,4,5,6,7)),
        yticklabels=(('pigeon', 'chimpanzee', 'capuchin', 'model', 'rhesus', 'dolphin', 'rat')))

    axes[1].scatter(np.array(model_halflives), 1*np.ones_like(model_halflives), s=2, color=palette[1])
    axes[1].scatter(np.median(model_halflives), 1, s=20, facecolor='none', edgecolor=palette[1])
    axes[1].scatter(np.array(pigeon_halflives), 2*np.ones_like(pigeon_halflives), s=2, color=palette[0])
    axes[1].scatter(pigeon_halflives_median, 2, s=20, facecolor='none', edgecolor=palette[0])
    axes[1].scatter(np.array(chimp_halflives), 3*np.ones_like(chimp_halflives), s=2, color=palette[0])
    axes[1].scatter(chimp_halflives_median, 3, s=20, facecolor='none', edgecolor=palette[0])
    axes[1].scatter(np.array(rhesus_halflives), 4*np.ones_like(rhesus_halflives), s=2, color=palette[0])
    axes[1].scatter(rhesus_halflives_median, 4, s=20, facecolor='none', edgecolor=palette[0])
    axes[1].scatter(np.array(rat_halflives), 5*np.ones_like(rat_halflives), s=2, color=palette[0])
    axes[1].scatter(rat_halflives_median, 5, s=20, facecolor='none', edgecolor=palette[0])
    axes[1].scatter(np.array(capuchin_halflives), 6*np.ones_like(capuchin_halflives), s=2, color=palette[0])
    axes[1].scatter(capuchin_halflives_median, 6, s=20, facecolor='none', edgecolor=palette[0])
    axes[1].scatter(np.array(dolphin_halflives), 7*np.ones_like(dolphin_halflives), s=2, color=palette[0])
    axes[1].scatter(dolphin_halflives_median, 7, s=20,  facecolor='none', edgecolor=palette[0])
    axes[1].set(xlabel="performance half-life (seconds)", ylim=((0.5, 7.5)), yticks=((1,2,3,4,5,6,7)),
        yticklabels=(('model', 'pigeon', 'chimp', 'rhesus', 'rat', 'capuchin', 'dolphin')))
    axes[1].set_xscale('log')
    axes[1].set_xticks([1, 10, 30, 60, 120, 300])
    axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    fig.savefig('plots/figures/forgetting_parameters.svg')
    fig.savefig('plots/figures/forgetting_parameters.pdf')

    print("model median baseline", np.median(model_baselines))
    print("model median half life", np.median(model_halflives))

seeds = [0,1,2,3,4,5,6,7,8,9]
# plot_euclidean_forgetting(seeds=seeds)
# fit_forgetting(seeds=seeds)
# fit_forgetting_individual(seeds=seeds)
# fit_forgetting_group(seeds=seeds, median_baseline=B, median_tau=tau)
plot_baseline_halflife(seeds=seeds)

# baseline(nCues=2, nSeeds=2, tTest=0.1, tGate=0.1)
# train(trainDA=0.0, seed=1, load=[], nTrain=10, tTrain=10)
# test(trainDA=0.0, testDA=0.0, seed=1, nCues=16, tTest=20, tGate=1, load='npz')
load_and_plot(nCues=8, seeds=[2,3,4], tTest=10, tGate=1, load='npz', trainDA=0.0, testDA=0.0)



