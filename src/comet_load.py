"""
Comet load.py
"""
import comet_ml
from comet_ml import API

comet_ml.config.save(api_key="57fHMWwvxUw6bvnjWLvRwSQFp")

comet_api = API()
workspace = "sdat2/6dactive"
found = False
for project in comet_api.get(workspace):
    if found:
        break
    print("    processing project", project, "...")
    print("        processing experiment", exp.id, end="")
    for exp in comet_api.get(workspace, project):
        print(".", end="")
        if exp.get_html() != None:
            print("\nFound html in %s!" % exp.url)
            found = True
            break


# python src/comet_load.py