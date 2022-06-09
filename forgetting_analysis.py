import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(context='paper', style='white', font='CMU Serif',
    rc={'font.size':10, 'mathtext.fontset': 'cm', 'axes.labelpad':0, 'axes.linewidth': 0.5})

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

def fit_forgetting_individual(seeds=[], tTest=20, tStart=0, tStep=1, trainDA=0.0, testDA=0.0):
	columns = ('seed', 'delay_length', 'correct', 'scaled')
	data = pd.read_pickle(f"data/20s/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
	tSamples = np.arange(tStart, tTest+tStep, tStep)
	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=((6,3)), sharey=True)
	# fig2, ax = plt.subplots(figsize=((6,3)))
	baselines = []
	taus = []
	half_lives = []
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
		# params, covariance = sp.optimize.curve_fit(exponential, t_monotonic, c_monotonic)
		# params, covariance = sp.optimize.curve_fit(exponential, t_left_aligned, c_monotonic)
		baseline = c_monotonic[0]
		def left_aligned_exponential(t, tau):
			return baseline * np.exp(-t/tau)
		params, covariance = sp.optimize.curve_fit(left_aligned_exponential, t_left_aligned, c_monotonic)
		# print(f"seed = {seed}")
		# print(f"baseline performance = {np.around(baseline, 1)}")
		# print(f"performance half life = {np.around(params[0]*np.log(2), 1)}")
		baselines.append(baseline)
		taus.append(params[0])
		half_lives.append(params[0]*np.log(2))
		ax.plot(tSamples[1:], corrects, label=seed)
		ax2.scatter(t_monotonic[0], left_aligned_exponential(t_left_aligned, params[0])[0], label=seed)
		ax2.plot(t_monotonic, left_aligned_exponential(t_left_aligned, params[0]), label=seed)
	ax.set(xlim=((0, tTest+tStep)), xticks=((0, tTest)), yticks=((0, 100)), ylim=((-10, 110)),
		ylabel=f'% Correct', xlabel='Delay Length (s)', title="raw simulated data")
	ax2.set(xlim=((0, tTest+tStep)), xticks=((0, tTest)), xlabel="Time (s)", title="best fit exponential")
	fig.tight_layout()
	# fig.legend(loc='upper right')
	fig.savefig(f'plots/DRT/forgetting_curves.pdf')

	# palette = sns.color_palette("dark")
	# fig2, ax2 = plt.subplots(figsize=((8.5, 1)))
	# x = np.array(baselines)
	# y = np.zeros_like(x)
	# ax2.axhline(0, color='k', alpha=0.2)
	# ax2.scatter(x, y, s=100, color=palette[1])
	# ax2.scatter(np.median(baselines), 0, s=150, facecolors="none", edgecolors=palette[1])
	# ax2.set(xlim=((50, 100)), xticks=((50, 100)))
	# fig2.savefig(f"plots/DRT/forgetting_curve_baselines.svg")

	# fig3, ax3 = plt.subplots(figsize=((3, 1)))
	# x = np.array(half_lives)
	# y = np.zeros_like(x)
	# ax3.axhline(0, color='k', alpha=0.2)
	# ax3.scatter(x, y, s=100, color=palette[1])
	# ax3.scatter(np.median(half_lives), 0, s=150, facecolors="none", edgecolors=palette[1])
	# ax3.set(xlim=((1, 10)), xticks=((1, 10)))
	# fig3.savefig(f"plots/DRT/forgetting_curve_halflives1.svg")

	# fig4, ax4 = plt.subplots(figsize=((2, 1)))
	# x = np.array(half_lives)
	# y = np.zeros_like(x)
	# ax4.axhline(0, color='k', alpha=0.2)
	# ax4.scatter(x, y, s=100, color=palette[1])
	# ax4.scatter(np.median(half_lives), 0, s=150, facecolors="none", edgecolors=palette[1])
	# ax4.set(xlim=((10, 30)), xticks=((10, 30)))
	# fig4.savefig(f"plots/DRT/forgetting_curve_halflives2.svg")

	print("baseline correct: \t", "min", np.around(np.min(baselines), 2), "max", np.around(np.max(baselines), 2), "mean", np.around(np.mean(baselines), 2), "median", np.around(np.median(baselines), 2))
	print("performance half life \t", "min", np.around(np.min(half_lives), 2), "max", np.around(np.max(half_lives), 2), "mean", np.around(np.mean(half_lives), 2), "median", np.around(np.median(half_lives), 2))

def fit_forgetting_group(seeds=[], tTest=20, tStart=0, tStep=1, trainDA=0.0, testDA=0.0, median_baseline=0, median_tau=0):
	columns = ('seed', 'delay_length', 'correct')
	data = pd.read_pickle(f"data/20s/DRT_trainDA={trainDA}_testDA={testDA}_allseeds.pkl")
	tSamples = np.arange(tStart, tTest+tStep, tStep)
	dfs = []
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
		for i in range(len(t_monotonic)):
			dfs.append(pd.DataFrame([[str(seed), t_monotonic[i], c_monotonic[i]]], columns=columns))
	df = pd.concat(dfs, ignore_index=True)
	# all_times = df['delay_length'].to_numpy().ravel()
	# all_corrects = df['correct'].to_numpy().ravel()
	# params, covariance = sp.optimize.curve_fit(exponential, all_times, all_corrects)
	# stds = np.sqrt(np.diag(covariance))
	# baseline_performance = params[0]
	# half_life = params[1] * np.log(2)
	# print(f"baseline performance = {np.around(params[0], 1)} +/- {np.around(stds[0],1)}")
	# print(f"performance tau = {np.around(params[1],1)} +/- {np.around(stds[1],1)}s")
	# print(f"performance half life = {np.around(params[1]*np.log(2),1)} +/- {np.around(stds[1]*np.log(2),1)}s")

	fig, ax = plt.subplots(figsize=((6,3)))
	sns.lineplot(data=df, x='delay_length', y='correct', label="simulated data")
	# ax.plot(tSamples, exponential(tSamples, params[0], params[1]), label="best fit exponential")
	ax.plot(tSamples, exponential(tSamples, median_baseline, median_tau), label="best fit exponential")
	sns.lineplot(data=df, x='delay_length', y='correct', hue="seed", alpha=0.2)
	ax.set(xlim=((0, tTest)), xticks=((0, tTest)), yticks=((0, 100)), ylim=((-10, 110)), ylabel=f'% Correct', xlabel='Delay Length (s)')
	fig.tight_layout()
	fig.legend(loc='upper right')
	fig.savefig(f'plots/DRT/forgetting_curve_group.pdf')
	fig.savefig(f'plots/DRT/forgetting_curve_group.svg')


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

seeds = [0,1,2,3,4,5,6,7,8,9]
# plot_euclidean_forgetting(seeds=seeds)
# fit_forgetting(seeds=seeds)
fit_forgetting_individual(seeds=seeds)
# fit_forgetting_group(seeds=seeds, median_baseline=B, median_tau=tau)