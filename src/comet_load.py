"""
Comet load.py
"""
import comet_ml
from comet_ml import API
import xarray as xr
from src.constants import FIGURE_PATH

comet_ml.config.save(api_key="57fHMWwvxUw6bvnjWLvRwSQFp")

comet_api = API()
print(comet_api.get())


def loop_through_project():
    workspace = "sdat2"  # /6dactive"
    project = "6dactive"

    for exp in comet_api.get(workspace, project):
        print("    processing project", project, "...")
        print("        processing experiment", exp.id, end="")
        print(".", end="")
        print(comet_api.get(workspace, project, exp.id))


def loop_through_experiment():
    workspace = "sdat2"  # /6dactive"
    project = "6dactive"
    # experiment = "f5e7f5b6d8c34f9b9c3d7e5a6d2f2c3"
    # experiment = "261f1786b8ab496e90170d593deba88f"
    for exp in comet_api.get(workspace, project):

        # exp = comet_api.get(workspace, project, experiment)
        metric_d = {}
        for metric, typ in [
            ("inum", int),
            ("anum", int),
            ("mae", float),
            ("rmse", float),
            ("r2", float),
        ]:
            metrics = exp.get_metrics(metric)
            metric_l = [typ(metrics[i]["metricValue"]) for i in range(len(metrics))]
            metric_d[metric] = (["point"], metric_l)
            print(exp.id, metric, "len(metrics)", len(metrics))
            # print("metrics", metrics)
            # for i in range(len(metrics)):
            #    print("metrics[" + str(i) + "]", metrics[i]["metricValue"])
        # print("metrics[0]", metrics[0]["metricValue"])
        ds = xr.Dataset(data_vars=metric_d)
        print(ds)


# loop_through_project()
loop_through_experiment()
# python src/comet_load.py
