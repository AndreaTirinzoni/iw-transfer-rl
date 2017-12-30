import numpy as np
import matplotlib.pyplot as plt

MARKERS = ["o", "D", "s", "X", "P", "*", "^"]
COLORS = ["#0e5ad3", "#bc2d14", "#22aa16", "#a011a3", "#d1ba0e", "#14ccc2", "#d67413"]
LINES = ["solid", "dashed", "dashdot", "dotted", "solid", "dashed", "dashdot", "dotted"]

def plot_curves(x_data, y_mean_data, y_std_data = None, title = "", x_label = "Episodes", y_label = "Performance", file_name = None):
    
    assert len(x_data) < 8
    
    plt.clf()
    plt.hold(True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    X = np.array(x_data)
    plt.xlim([X.min(),X.max()])
    
    for i in range(len(x_data)):
        
        plt.plot(x_data[i],y_mean_data[i], linewidth = 3, color = COLORS[i], marker = MARKERS[i], markersize = 8.0, linestyle = LINES[i])
        if y_std_data is not None:
            plt.fill_between(x_data[i], y_mean_data[i] - y_std_data[i], y_mean_data[i] + y_std_data[i], facecolor = COLORS[i], edgecolor = COLORS[i], alpha = 0.5)
    
    plt.show()
    

def extract_steps_data(steps, names):
    
    n_names = len(names)
    data = [[] for _ in range(n_names)]
    
    for step in steps:
        for i in range(n_names):
            data[i].append(step[names[i]])
    
    return [np.array(d) for d in data]

def extract_runs_data(runs, names):
    
    n_names = len(names)
    data = [[] for _ in range(n_names)]
    
    for run in runs:
        steps = extract_steps_data(run["steps"], names)
        for i in range(n_names):
            data[i].append(steps[i])
    
    return [np.array(d) for d in data]

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
            run_y_mean.append(step[y_name])
    
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
    