import nengo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def plotActivities(times, aEns, aTarA, network, neuron_type, ens, nT, nTrain):
    if nT==0 or (nT+1)==nTrain:
        for n in range(aTarA.shape[1]):
            ymax1 = np.max(aEns[:,n])
            ymax2 = np.max(aTarA[:,n])
            ymax = np.max([ymax1, ymax2]) if ymax1+ymax2>0 else 1
            fig, ax = plt.subplots(figsize=((6, 2)))
            ax.plot(times, aTarA[:,n], alpha=0.5, label='target')
            ax.plot(times, aEns[:,n], alpha=0.5, label=neuron_type)
            ax.set(xlabel='time (s)', ylabel=r"$a(t)$ (Hz)",
                # xlim=((0, times[-1])), xticks=((0, times[-1])),
                xlim=((0, times[-1])), ylim=((0, ymax)), xticks=((0, times[-1])), yticks=((0, ymax)))
            plt.legend(loc='upper right')
            sns.despine()
            plt.tight_layout()
            plt.savefig(f'plots/{network}/{neuron_type}/{ens}/{nT+1}p{nTrain}_{n}.pdf')
            plt.close('all')

def checkTrain(times, aEns, aTar, stage):
    for n in range(aTar.shape[1]):
        ymax1 = np.max(aEns[:,n])
        ymax2 = np.max(aTar[:,n])
        ymax = np.max([ymax1, ymax2]) if ymax1+ymax2>0 else 1
        fig, ax = plt.subplots(figsize=((6, 2)))
        ax.plot(times, aEns[:,n], alpha=0.5)
        ax.plot(times, aTar[:,n], alpha=0.5, color='gray')
        ax.set(xlabel='time (s)', ylabel=r"$a(t)$ (Hz)", xlim=((0, times[-1])), ylim=((0, ymax)), xticks=((0, times[-1])), yticks=((0, ymax)))
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'plots/DRT/training/stage{stage}_neuron{n}.pdf')
        plt.close('all')