from trlib.experiments.results import Result
from trlib.experiments.visualization import plot_average

fqi = Result.load_json("fqi.json")
laroche2017 = Result.load_json("laroche2017.json")
lazaric2008 = Result.load_json("lazaric2008.json")
wfqi_ideal = Result.load_json("wfqi-ideal.json")
wfqi_mean = Result.load_json("wfqi-mean.json")
#wfqi_heuristic = Result.load_json("wfqi-heuristic.json")

plot_average([fqi, laroche2017, lazaric2008, wfqi_ideal, wfqi_mean], "n_episodes", "perf_disc_greedy_mean", names = ["FQI", "SDT", "RBT", "WFQI-ID", "WFQI"], file_name = "perf_greedy")