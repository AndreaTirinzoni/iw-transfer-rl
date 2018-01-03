from trlib.experiments.results import Result
from trlib.experiments.visualization import plot_average

fqi = Result.load_json("fqi.json")
laroche2017 = Result.load_json("laroche2017.json")
lazaric2008 = Result.load_json("lazaric2008.json")

plot_average([fqi, laroche2017, lazaric2008], "n_episodes", "perf_disc_mean", names = ["FQI", "Laroche2017", "Lazaric2008"], file_name = "perf")
plot_average([fqi, laroche2017, lazaric2008], "n_episodes", "perf_disc_greedy_mean", names = ["FQI", "Laroche2017", "Lazaric2008"], file_name = "perf_greedy")