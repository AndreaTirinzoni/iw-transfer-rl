import numpy as np
import matplotlib.pyplot as plt

def plot_steps(result, x_name = "n_episodes", y_name = "perf_disc"):
    
    x = []
    y_mean = []
    y_std = []
    
    for step in result.steps:
        x.append(step[x_name])
        y_mean.append(step[y_name][0])
        y_std.append(step[y_name][1])
    
    x = np.array(x)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    
    plt.clf()
    plt.hold(True)
    
    plt.title(result.algorithm)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim([np.min(x),np.max(x)])
    
    plt.plot(x,y_mean, linewidth = 3, color = "#1864dd", marker = "D", markersize = 8.0)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, facecolor = "#1864dd", edgecolor = "#1864dd", alpha = 0.5)
    
    plt.show()
    
def plot_experiment(result, x_name = "n_episodes", y_name = "perf_disc"):
    
    means = []
    
    for run in result.runs:
        x = []
        run_y_mean = []
        
        for step in run["steps"]:
            x.append(step[x_name])
            run_y_mean.append(step[y_name][0])
    
        x = np.array(x)
        means.append(run_y_mean)
    
    means = np.array(means).T
    y_mean = np.mean(means,1)
    y_std = np.std(means,1) / np.sqrt(result.n_runs)
    
    plt.clf()
    plt.hold(True)
    
    plt.title(result.experiment)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xlim([np.min(x),np.max(x)])
    
    plt.plot(x,y_mean, linewidth = 3, color = "#1864dd", marker = "D", markersize = 8.0)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, facecolor = "#1864dd", edgecolor = "#1864dd", alpha = 0.5)
    
    plt.show()
    