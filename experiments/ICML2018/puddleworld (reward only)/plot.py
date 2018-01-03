from trlib.experiments.results import Result
from trlib.experiments.visualization import plot_average

fqi = Result.load_json("fqi.json")
laroche2017 = Result.load_json("laroche2017.json")
lazaric2008 = Result.load_json("lazaric2008.json")
wfqi_ideal = Result.load_json("wfqi-ideal.json")

plot_average([fqi, laroche2017, lazaric2008, wfqi_ideal], "n_episodes", "perf_disc_mean", names = ["FQI", "Laroche2017", "Lazaric2008", "WFQI-ID"], file_name = "perf")
plot_average([fqi, laroche2017, lazaric2008, wfqi_ideal], "n_episodes", "perf_disc_greedy_mean", names = ["FQI", "Laroche2017", "Lazaric2008", "WFQI-ID"], file_name = "perf_greedy")