import nengo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def plotActivities(times, aEns, aTarA, network, neuron_type, nT, nTrain):
    for n in range(aTarA.shape[1]):
        fig, ax = plt.subplots(figsize=((6, 2)))
        ax.plot(times, aTarA[:,n], alpha=0.5, label='ReLu (target)')
        ax.plot(times, aEns[:,n], alpha=0.5, label=neuron_type)
        ax.set(xlabel='time (s)', ylabel=r"$a(t)$ (Hz)",
        	xlim=((0, times[-1])), ylim=((0, 35)), xticks=((0, times[-1])), yticks=((0, 35)))
        plt.legend(loc='upper right')
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'plots/{network}/tuning/{neuron_type}/{nT}p{nTrain}_{n}.pdf')
        plt.close('all')