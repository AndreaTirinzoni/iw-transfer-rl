from trlib.experiments.results import Result
from trlib.experiments.visualization import plot_average

# Load results
fqi = Result.load_json("fqi.json")
laroche2017 = Result.load_json("laroche2017.json")
lazaric2008 = Result.load_json("lazaric2008.json")
wfqi_ideal = Result.load_json("wfqi-ideal.json")
wfqi_mean = Result.load_json("wfqi-mean.json")

# Plot results
y_name = 'perf_disc_greedy_mean'  # plot discounted return
# y_name = 'perf_avg_greedy_mean'  # plot average return
# y_name = 'perf_disc_greedy_steps'  # plot average duration of each episode
plot_average([fqi, laroche2017, lazaric2008, wfqi_ideal, wfqi_mean], x_name="n_episodes",
             y_name=y_name, names=["FQI", "SDT", "RBT", "WFQI-ID", "WFQI"], file_name="perf")
