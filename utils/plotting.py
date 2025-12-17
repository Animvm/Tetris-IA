import matplotlib.pyplot as plt
import numpy as np
import os

# grafica metricas de entrenamiento con promedios moviles
def plot_metrics(rewards, lines, window=50, savepath=None, title="Training progress"):
    episodes = np.arange(1, len(rewards)+1)
    rewards = np.array(rewards)
    lines = np.array(lines)

    # calcula promedio movil
    def moving_avg(x, w):
        if len(x) < w:
            return np.convolve(x, np.ones(len(x))/len(x), mode='valid')
        return np.convolve(x, np.ones(w)/w, mode='valid')

    fig, axs = plt.subplots(2,1, figsize=(8,6), tight_layout=True)
    axs[0].plot(episodes, rewards, alpha=0.3, label='reward per ep')
    mav = moving_avg(rewards, window)
    axs[0].plot(np.arange(len(mav))+1, mav, label=f'mavg{window}')
    axs[0].set_ylabel('Reward')
    axs[0].legend()

    axs[1].plot(episodes, lines, alpha=0.3, label='lines per ep')
    mavl = moving_avg(lines, window)
    axs[1].plot(np.arange(len(mavl))+1, mavl, label=f'mavg{window}')
    axs[1].set_ylabel('Lines')
    axs[1].set_xlabel('Episode')
    axs[1].legend()

    fig.suptitle(title)
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
    plt.show()
