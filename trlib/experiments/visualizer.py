import numpy as np
import matplotlib.pyplot as plt

def plot_performance(result):
    
    episodes = []
    means = []
    stds = []
    
    for step in result.steps:
        episodes.append(step["n_episodes"])
        means.append(step["perf"][0])
        stds.append(step["perf"][1])
    
    episodes = np.array(episodes)
    means = np.array(means)
    stds = np.array(stds)
    
    plt.clf()
    plt.hold(True)
    
    plt.title(result.algorithm)
    plt.xlabel("Episodes")
    plt.ylabel("Performance")
    plt.xlim([np.min(episodes),np.max(episodes)])
    
    plt.plot(episodes,means, linewidth = 3, color = "#1864dd", marker = "D", markersize = 8.0)
    plt.fill_between(episodes, means - stds, means + stds, facecolor = "#1864dd", edgecolor = "#1864dd", alpha = 0.5)
    
    plt.show()
    