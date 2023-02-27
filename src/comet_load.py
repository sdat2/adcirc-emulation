"""
Comet load.py
"""
import comet_ml
from comet_ml import API

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
    experiment = "261f1786b8ab496e90170d593deba88f"

    exp = comet_api.get(workspace, project, experiment)
    for metric in ["mae", "rmse", "r2", "inum", "anum"]:
        metrics = exp.get_metrics(metric)
        print("len(metrics)", len(metrics))
        print("metrics", metrics)
        for i in range(len(metrics)):
            print("metrics[" + str(i) + "]", metrics[i]["metricValue"])
    # print("metrics[0]", metrics[0]["metricValue"])


# loop_through_project()
loop_through_experiment()
# python src/comet_load.py
