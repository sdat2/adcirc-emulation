"""
Active learning plots.
"""
import os
import xarray as xr
import matplotlib.pyplot as plt
from sithom.plot import plot_defaults, lim, label_subplots

path: str = "/work/n01/n01/sithom/new-orleans/data/emulation_angle_pos_newei"
ds_list = []

for i in range(100, 130):
    file_name = "plotting_data" + str(i) + ".nc"
    file_path = os.path.join(path, file_name)
    ds = xr.open_dataset(file_path).expand_dims({"t": [i]})
    ds_list.append(ds)

    print(ds)
    # python src/plot/active.py

ds = xr.merge(ds_list)
print(ds)
