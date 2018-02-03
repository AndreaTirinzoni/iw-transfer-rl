import numpy as np
import matplotlib.pyplot as plt

MARKERS = ["o", "D", "s", "^", "v", "p", "*"]
COLORS = ["#0e5ad3", "#bc2d14", "#22aa16", "#a011a3", "#d1ba0e", "#14ccc2", "#d67413"]
LINES = ["solid", "dashed", "dashdot", "dotted", "solid", "dashed", "dashdot", "dotted"]

def plot_curves(x_data, y_mean_data, y_std_data = None, title = "", x_label = "Episodes", y_label = "Performance", names = None, file_name = None):
    
    assert len(x_data) < 8
    
    plt.style.use('ggplot')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    #plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['figure.titlesize'] = 20
    
    fig, ax = plt.subplots()
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    X = np.array(x_data)
    plt.xlim([X.min(),X.max()])
    
    for i in range(len(x_data)):
        
        ax.plot(x_data[i],y_mean_data[i], linewidth = 4, color = COLORS[i], marker = MARKERS[i], markersize = 8.0, linestyle = LINES[i], label = names[i] if names is not None else None)
        if y_std_data is not None:
            ax.fill_between(x_data[i], y_mean_data[i] - y_std_data[i], y_mean_data[i] + y_std_data[i], facecolor = COLORS[i], edgecolor = COLORS[i], alpha = 0.3)
    
    if names is not None:
        ax.legend(loc='best', numpoints = 1, fancybox=True, frameon = False)
        
    if file_name is not None:
        plt.savefig(file_name + ".pdf", format='pdf')
    
    plt.show()

def plot_average(results, x_name, y_name, title = "", x_label = "Episodes", y_label = "Performance", names = None, file_name = None):
    
    x = []
    y_mean = []
    y_std = []
    
    for result in results:
        runs = extract_runs_data(result.runs, [x_name, y_name])
        x.append(np.mean(runs[0].T,1))
        y_mean.append(np.mean(runs[1].T,1))
        y_std.append(np.std(runs[1].T,1) / np.sqrt(runs[1].shape[0]))
        
    plot_curves(x, y_mean, y_std, title = title, x_label = x_label, y_label = y_label, names = names, file_name = file_name)

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

    plt.show()