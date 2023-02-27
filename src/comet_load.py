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


loop_through_project()
# python src/comet_load.py
