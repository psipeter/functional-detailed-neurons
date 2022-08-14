import numpy as np
import nengo
import neuron
from scipy.integrate import solve_ivp
from nengo.params import Default, NumberParam
from nengo.builder import Builder, Operator, Signal
from nengo.builder.neurons import NeuronType, SimNeurons
from nengo.builder.connection import BuiltConnection
from nengolib.signal import LinearSystem
import warnings

class AMPA(nengo.synapses.Synapse):
	def __init__(self): super().__init__()
		
class GABA(nengo.synapses.Synapse):
	def __init__(self): super().__init__()

class NMDA(nengo.synapses.Synapse):
	def __init__(self): super().__init__()

class LIF(NeuronType):

	probeable = ("spikes", "voltage", "refractory_time")

	def __init__(self, tau_rc=0.05, tau_ref=0.005):
		super(LIF, self).__init__()
		self.tau_rc = tau_rc
		self.tau_ref = tau_ref

	def gain_bias(self, max_rates, intercepts):
		return np.ones_like(max_rates), np.zeros_like(intercepts)

	def max_rates_intercepts(self, gain, bias):
		return np.zeros_like(gain), np.zeros_like(bias)

	def step_math(self, dt, J, spiked, voltage, refractory_time):
		refractory_time -= dt
		delta_t = (dt - refractory_time).clip(0, dt)
		voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)
		spiked_mask = voltage > 1
		spiked[:] = spiked_mask / dt
		t_spike = dt + self.tau_rc * np.log1p(-(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1))
		voltage[voltage < 0] = 0
		voltage[spiked_mask] = 0
		refractory_time[spiked_mask] = self.tau_ref + t_spike

@Builder.register(LIF)
def build_lif(model, lif, neurons):
	model.sig[neurons]['voltage'] = Signal(np.zeros(neurons.size_in), name="%s.voltage" % neurons)
	model.sig[neurons]['refractory_time'] = Signal(np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
	model.add_op(SimNeurons(
		neurons=lif,
		J=model.sig[neurons]['in'],
		output=model.sig[neurons]['out'],
		states=[model.sig[neurons]['voltage'],
			model.sig[neurons]['refractory_time']]))

class Izhikevich(NeuronType):
	'''
	This implementation is based on the original paper
		E. M. Izhikevich, "Simple model of spiking neurons."
		IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572.
		(http://www.izhikevich.org/publications/spikes.pdf)
	What was originally 'v' we term 'voltage', which represents the membrane potential of each neuron.
	What was originally 'u' we term 'recovery', hich represents membrane recovery, "which accounts for the activation
	of K+ ionic currents and inactivation of Na+ ionic currents."
	The 'a', 'b', 'c', and 'd' parameters are also renamed

	We use default values that correspond to regular spiking ('RS') neurons.
	For other classes of neurons, set the parameters as follows.
		* Intrinsically bursting (IB): ``reset_voltage=-55, reset_recovery=4``
		* Chattering (CH): ``reset_voltage=-50, reset_recovery=2``
		* Fast spiking (FS): ``tau_recovery=0.1``
		* Low-threshold spiking (LTS): ``coupling=0.25``
		* Resonator (RZ): ``tau_recovery=0.1, coupling=0.26``
	'''

	probeable = ("spikes", "voltage", "recovery")
	threshold = NumberParam('threshold')  # spike threshold
	tau_recovery = NumberParam("tau_recovery", low=0, low_open=True)
	coupling = NumberParam("coupling", low=0)
	reset_voltage = NumberParam("reset_voltage")
	reset_recovery = NumberParam("reset_recovery")

	def __init__(self, threshold=-40, tau_recovery=0.02, coupling=0.2, reset_voltage=-65.0, reset_recovery=8.0):
		super(Izhikevich, self).__init__()
		self.threshold = threshold
		self.tau_recovery = tau_recovery
		self.coupling = coupling
		self.reset_voltage = reset_voltage
		self.reset_recovery = reset_recovery

	def gain_bias(self, max_rates, intercepts):
		return np.ones_like(max_rates), np.zeros_like(intercepts)

	def max_rates_intercepts(self, gain, bias):
		return np.zeros_like(gain), np.zeros_like(bias)

	def step_math(self, dt, J, spiked, voltage, recovery):
		dV = (0.04 * voltage ** 2 + 5 * voltage + 140 - recovery + J) * 1000
		voltage[:] += dV * dt
		spiked[:] = (voltage >= self.threshold) / dt
		voltage[spiked > 0] = self.reset_voltage
		dU = (self.tau_recovery * (self.coupling * voltage - recovery)) * 1000
		recovery[:] += dU * dt
		recovery[spiked > 0] = recovery[spiked > 0] + self.reset_recovery

@Builder.register(Izhikevich)
def build_izhikevich(model, lif, neurons):
	model.sig[neurons]['voltage'] = Signal(np.zeros(neurons.size_in), name="%s.voltage" % neurons)
	model.sig[neurons]['recovery'] = Signal(np.zeros(neurons.size_in), name="%s.recovery" % neurons)
	model.add_op(SimNeurons(
		neurons=lif,
		J=model.sig[neurons]['in'],
		output=model.sig[neurons]['out'],
		states=[model.sig[neurons]['voltage'],
			model.sig[neurons]['recovery']]))


class Wilson(NeuronType):
	'''
	An intermediate-complexity neuron developed by Wilson in "Simplified dynamics of human and mammalian neocortical neurons (1999)"
	Extends the FitzHugo-Nagumo equations to incorporate electrophysiological detail,
	including Ohm's Law and equilibrium potentials of four ionic currents in neocortical neurons (K, Na, R, AHP).
	The resulting model consists of three coupled ODEs representing voltage, conductance, and recovery,
	can generate realistic action potentials, and naturally produces adaptation, bursting, and other neocortical behaviors.
	Due to the lower number of equations (and cubic dynamics of each equation),
	simulation is relatively fast and certain analytical characterizations are still possible.
	'''
	probeable = ('spikes', 'voltage', 'recovery', 'conductance', 'AP')
	threshold = NumberParam('threshold')  # spike threshold
	tauV = NumberParam('tauV')  # time constant
	tauR = NumberParam('tauR')  # time constant
	tauH = NumberParam('tauH')  # time constant
	V0 = -0.754  # initial voltage
	R0 = 0.279  # initial recovery
	H0 = 0  # initial conductance

	def __init__(self, threshold=-0.4, tauV=0.00097, tauR=0.0056, tauH=0.0990):
		super(Wilson, self).__init__()
		self.threshold = threshold
		self.tauV = tauV
		self.tauR = tauR
		self.tauH = tauH

	def gain_bias(self, max_rates, intercepts):
		return np.ones_like(max_rates), np.zeros_like(intercepts)

	def max_rates_intercepts(self, gain, bias):
		return np.zeros_like(gain), np.zeros_like(bias)

	def step_math(self, dt, J, spiked, V, R, H, AP, dtOde=1e-4):
		# Use Euler's Method with a smaller dt to simulate the neural dynamics
		for t in range(int(dt/dtOde)):
			dV = -(17.81 + 47.58*V + 33.80*np.square(V))*(V-0.48) - 26*R*(V+0.95) - 13*H*(V+0.95) + J
			dR = -R + 1.29*V + 0.79 + 3.30*np.square(V+0.38)
			dH = -H + 11*(V+0.754)*(V+0.69)
			V[:] += (dV*dtOde/self.tauV).clip(-0.8, 0.4)
			R[:] += dR*dtOde/self.tauR
			H[:] += dH*dtOde/self.tauH
		spiked[:] = ((V > self.threshold) & (~AP)) / dt
		AP[:] = V > self.threshold
		return spiked, V, R, H, AP


@Builder.register(Wilson)
def build_wilson(model, neuron_type, neurons):
	model.sig[neurons]['voltage'] = Signal(neuron_type.V0*np.ones(neurons.size_in), name="%s.voltage" % neurons)
	model.sig[neurons]['recovery'] = Signal(neuron_type.R0*np.ones(neurons.size_in), name="%s.recovery" % neurons)
	model.sig[neurons]['conductance'] = Signal(neuron_type.H0*np.ones(neurons.size_in), name="%s.conductance" % neurons)
	model.sig[neurons]['AP'] = Signal(np.zeros(neurons.size_in, dtype=bool), name="%s.AP" % neurons)
	model.add_op(SimNeurons(neurons=neuron_type, J=model.sig[neurons]['in'], output=model.sig[neurons]['out'],
		states=[model.sig[neurons]['voltage'], model.sig[neurons]['recovery'], model.sig[neurons]['conductance'], model.sig[neurons]['AP']]))



class NEURON(NeuronType):

	'''
	Reproduced from Durstewitz, Seamans, and Sejnowski "Dopamine-Mediated Stabilization of Delay-Period Activity in a Network Model of Prefrontal Cortex (2000)"
	of pyramidal neurons that includes four compartments (soma, proximal-, distal-, and basal-dendrites)
	and six ionic currents (two for sodium, three for potassium, and one for calcium).
	The Durstewitz reconstruction accurately reproduces electrophysiological recordings from layer-V intrinsically-bursting pyramidal neurons in rat PFC,
	cells that are known to be active during the delay period of working memory tasks.
	Also included is an inhibitory interneuron that includes two compartments (soma, dendrites), similar channels, and similar validation 
	This neuron model is implemented in NEURON and uses conductance-based synapses, distributed randomly on the three dendritic compartments.
	'''

	probeable = ('spikes', 'voltage')
	threshold = NumberParam('threshold')  # spike threshold
	dtNeuron = NumberParam('dtNeuron')  # time constant
	DA = NumberParam('DA')

	def __init__(self, cell_type, threshold=-40, DA=0, dtNeuron=0.1):
		super(NEURON, self).__init__()
		self.cell_type = cell_type
		self.threshold = threshold
		self.dtNeuron = dtNeuron
		self.DA = DA

	def gain_bias(self, max_rates, intercepts):
		return np.ones_like(max_rates), np.zeros_like(intercepts)

	def max_rates_intercepts(self, gain, bias):
		return np.zeros_like(gain), np.zeros_like(bias)

	def step_math(self, V, spiked, t, dt, neurons, v_recs, spk_vecs, spk_recs, spk_before):
		if neuron.h.t < t*1000:  # Nengo starts at t=dt
			neuron.h.tstop = t*1000
			neuron.h.continuerun(neuron.h.tstop)  # NEURON simulation, advances all cells in NEURON's memory to tstop
		spk_after = [list(spk_vecs[n]) for n in range(len(neurons))]
		for n in range(len(neurons)):
			V[n] = v_recs[n][-1]
			spiked[n] = (len(spk_after[n]) - len(spk_before[n])) / dt
			spk_before[n] = list(spk_after[n])


@Builder.register(NEURON)
def build_hoc(model, neuron_type, neurons):
	model.sig[neurons]['voltage'] = Signal(np.zeros(neurons.size_in), name="%s.voltage"%neurons)
	neuron.h.load_file('stdrun.hoc')
	neuron.h.load_file('NEURON/cells.hoc')
	neuronop = SimNEURON(
		neuron_type=neuron_type,
		n_neurons=neurons.size_in,
		J=model.sig[neurons]['in'],
		output=model.sig[neurons]['out'],
		states=[model.time, model.sig[neurons]['voltage']],
		dt=model.dt)
	model.params[neurons] = neuronop.neurons
	model.add_op(neuronop)

# The operator object for Nengo, which stores lists of NEURON cells, spike events, and voltages,
# and transmits these lists to the neuron object above at every timestep
class SimNEURON(Operator):

	def __init__(self, neuron_type, n_neurons, J, output, states, dt):
		super(SimNEURON, self).__init__()
		self.neuron_type = neuron_type
		rng = np.random.RandomState(seed=0)
		rGeos = rng.normal(1, 0.2, size=(n_neurons,))
		rCms = rng.normal(1, 0.05, size=(n_neurons,))
		rRs = rng.normal(1, 0.1, size=(n_neurons,))
		v0s = rng.uniform(-80, -60, size=(n_neurons, ))
		self.neurons = []
		for n in range(n_neurons):
			if self.neuron_type.cell_type=='Pyramidal':
				self.neurons.append(neuron.h.Pyramidal(rGeos[n], rCms[n], rRs[n], self.neuron_type.DA))
			elif self.neuron_type.cell_type=='Interneuron':
				self.neurons.append(neuron.h.Interneuron(rGeos[n], rCms[n], rRs[n], self.neuron_type.DA))
		self.reads = [states[0], J]
		self.sets = [output, states[1]]
		self.updates = []
		self.incs = []
		self.v_recs = []
		self.spk_vecs = []
		self.spk_recs = []
		self.spk_before = [[] for n in range(n_neurons)]
		for n in range(n_neurons):
			self.v_recs.append(neuron.h.Vector())
			self.v_recs[n].record(self.neurons[n].soma(0.5)._ref_v)
			self.spk_vecs.append(neuron.h.Vector())
			self.spk_recs.append(neuron.h.APCount(self.neurons[n].soma(0.5), self.neuron_type.threshold))
			self.spk_recs[n].record(neuron.h.ref(self.spk_vecs[n]))
			self.neurons[n].set_v(v0s[n])
		neuron.h.dt = self.neuron_type.dtNeuron
		neuron.h.tstop = 0

	def make_step(self, signals, dt, rng):
		J = signals[self.current]
		output = signals[self.output]
		voltage = signals[self.voltage]
		time = signals[self.time]
		def step_nrn():
			self.neuron_type.step_math(voltage, output, time, dt, self.neurons, self.v_recs, self.spk_vecs, self.spk_recs, self.spk_before)
		return step_nrn

	@property
	def time(self):
		return self.reads[0]
	@property
	def current(self):
		return self.reads[1]
	@property
	def output(self):
		return self.sets[0]
	@property
	def voltage(self):
		return self.sets[1]

# Transmits spikes between Nengo's signal (an array of spikes) and NEURON's NetCons
# (which trigger spike events on the cell's synapses)
# Also update the weight value on the NetCon objects in time for a spike event
class NrnConnect(Operator):

	def __init__(self, conn, neurons, netcons, spikes, states, dt):
		super(NrnConnect, self).__init__()
		self.conn = conn
		self.neurons = neurons
		self.dt = dt
		self.time = states[0]
		self.reads = [spikes, states[0]]
		self.updates = []
		self.sets = []
		self.incs = []
		self.netcons = netcons

	def make_step(self, signals, dt, rng):
		spikes = signals[self.spikes]
		time = signals[self.time]
		def step():
			t_neuron = time.item()*1000
			for pre in range(spikes.shape[0]):
				if spikes[pre] > 0:
					for post in range(len(self.neurons)):
						self.netcons[pre, post].weight[0] = np.abs(self.conn.w[pre, post])  #  update weight
						self.netcons[pre, post].syn().e = 0 if self.conn.w[pre, post] > 0 else -70  #  update reversal potential
						self.netcons[pre, post].event(t_neuron)  # deliver spike
		return step

	@property
	def spikes(self):
		return self.reads[0]


# Define a connection between a presynaptic ensemble and the ensemble populated by NEURON cells
# Requires storing many lists in memory, including NEURON's synapse objects and NetCon objects
# Overrides Nengo's default build_connection, so we include a statement to build normally if the ensemble is not from NEUROn
@Builder.register(nengo.Connection)
def build_connection(model, conn):
	if isinstance(conn.post_obj, nengo.Ensemble) and isinstance(conn.post_obj.neuron_type, NEURON):
		assert isinstance(conn.pre_obj, nengo.Ensemble)
		assert isinstance(conn.solver, nengo.solvers.NoSolver)
		assert 'spikes' in conn.pre_obj.neuron_type.probeable
		post_obj = conn.post_obj
		pre_obj = conn.pre_obj
		model.sig[conn]['in'] = model.sig[pre_obj]['out']
		special_synapse = False
		if isinstance(conn.synapse, AMPA): special_synapse = "AMPA"  # special named synapses
		elif isinstance(conn.synapse, GABA): special_synapse = "GABA"
		elif isinstance(conn.synapse, NMDA): special_synapse = "NMDA"
		else:
			assert isinstance(conn.synapse, LinearSystem)
			taus = -1.0/np.array(conn.synapse.poles) * 1000  # normal synapse

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")        
			conn.rng = np.random.RandomState(model.seeds[conn])
			conn.w = conn.solver.values
			conn.locations = conn.rng.uniform(0, 1, size=(pre_obj.n_neurons, post_obj.n_neurons))
			conn.synapses = np.zeros((pre_obj.n_neurons, post_obj.n_neurons), dtype=list)
			conn.netcons = np.zeros((pre_obj.n_neurons, post_obj.n_neurons), dtype=list)
			conn.compartments = conn.rng.randint(0, 3, size=(pre_obj.n_neurons, post_obj.n_neurons))
			conn.v_recs = []

		for post in range(post_obj.n_neurons):
			nrn = model.params[post_obj.neurons][post]
			for pre in range(pre_obj.n_neurons):
				if conn.post_obj.neuron_type.cell_type=='Pyramidal':
					if conn.compartments[pre, post] == 0:
						loc = nrn.prox(conn.locations[pre, post])
					elif conn.compartments[pre, post] == 1:
						loc = nrn.dist(conn.locations[pre, post])
					else:
						loc = nrn.basal(conn.locations[pre, post])
				elif conn.post_obj.neuron_type.cell_type=='Interneuron':
					loc = nrn.dendrite(conn.locations[pre, post])
				if special_synapse:
					if special_synapse == "AMPA": syn = neuron.h.ampa(loc)
					elif special_synapse == "GABA": syn = neuron.h.gaba(loc)
					elif special_synapse == "NMDA": syn = neuron.h.nmda(loc)
					syn.DA = conn.post_obj.neuron_type.DA
				elif len(taus) == 1:
					syn = neuron.h.ExpSyn(loc)
					syn.tau = taus[0]
				elif len(taus) == 2:
					syn = neuron.h.Exp2Syn(loc)
					syn.tau1 = np.min(taus)
					syn.tau2 = np.max(taus)
				conn.synapses[pre, post] = syn
				conn.netcons[pre, post] = neuron.h.NetCon(None, conn.synapses[pre, post])
				conn.netcons[pre, post].weight[0] = np.abs(conn.w[pre, post])
				conn.netcons[pre, post].syn().e = 0 if conn.w[pre, post] > 0 else -70
			conn.v_recs.append(neuron.h.Vector())
			conn.v_recs[post].record(nrn.soma(0.5)._ref_v)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")  
			conn.nrnConnect = NrnConnect(
				conn,
				model.params[post_obj.neurons],
				conn.netcons,
				model.sig[conn.pre_obj]['out'],
				states=[model.time],
				dt=model.dt)
		model.add_op(conn.nrnConnect)
		model.params[conn] = BuiltConnection(eval_points=None, solver_info=None, transform=None, weights=None)

	# Otherwise, build the connection normally
	else:
		c = nengo.builder.connection.build_connection(model, conn)
		model.sig[conn]['weights'].readonly = False
		return c

# Method for cleaning up all NEURON objects in memory after a Nengo simulation
def nrnReset(sim, model):
	for key in list(sim.model.params.keys()):
		if type(key) == nengo.ensemble.Neurons:
			del(sim.model.params[key])
	for op in sim.model.operators:
		if isinstance(op, SimNEURON):
			for v_rec in op.v_recs:
				v_rec.play_remove()
			for spk_vec in op.spk_vecs:
				spk_vec.play_remove()
			del(op.neurons)
		if isinstance(op, NrnConnect):
			del(op.neurons)
			del(op.netcons)
	for conn in model.connections:
		if hasattr(conn, 'v_recs'):
			for v_rec in conn.v_recs:
				v_rec.play_remove()
		if hasattr(conn, 'synapses'):
			del(conn.synapses)
		if hasattr(conn, 'netcons'):
			del(conn.netcons)
		if hasattr(conn, 'nrnConnect'):
			del(conn.nrnConnect)



class PoissonSpikingReLU(NeuronType):

	probeable = ('spikes')
	
	def __init__(self, seed):
		super(PoissonSpikingReLU, self).__init__()
		self.rng = np.random.RandomState(seed=seed)
		self.amplitude = 1

	def gain_bias(self, max_rates, intercepts):
		"""Determine gain and bias by shifting and scaling the lines."""
		max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
		intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)
		gain = max_rates / (1 - intercepts)
		bias = -intercepts * gain
		return gain, bias

	def max_rates_intercepts(self, gain, bias):
		"""Compute the inverse of gain_bias."""
		intercepts = -bias / gain
		max_rates = gain * (1 - intercepts)
		return max_rates, intercepts

	def step_math(self, dt, J, output, *states):
		# Note: J is the desired output rate, not the input current
		output[...] = np.maximum(0.0,  (self.amplitude / dt) * self.rng.poisson(np.abs(J) * dt, output.size)* np.sign(J))