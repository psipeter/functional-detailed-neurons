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
from utils import LearningNode, trainD, fitSinusoid
from plotter import checkTrain
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
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


# baseline(nCues=2, nSeeds=2, tTest=0.1, tGate=0.1)
# train(trainDA=0.0, seed=1, load=[], nTrain=10, tTrain=10)
# test(trainDA=0.0, testDA=0.0, seed=1, nCues=16, tTest=20, tGate=1, load='npz')
load_and_plot(nCues=8, seeds=[2,3,4], tTest=10, tGate=1, load='npz', trainDA=0.0, testDA=0.0)