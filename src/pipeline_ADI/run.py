from contrast_pipeline.post_processing import run_pipeline, load_config
import copy
import yaml
import numpy as np
from contrast_pipeline.plotting import plot_all_best


config = load_config(
    "/home/aosimul/noah/src/pipeline_ADI/configs/ghost.yaml",
    defaults_path="/home/aosimul/noah/src/pipeline_ADI/configs/default_values.yaml",
)

new_config = copy.deepcopy(config)


# for method in ["gram", "lowrank"]:
#     new_config["fake_planet"]["pca_method"] = method
#     if method == "lowrank":
#         for iter in [1, 2]:
#             new_config["fake_planet"]["niter"] = iter
#             for trunc in [10, 50, None]:
#                 new_config["fake_planet"]["approx_svd_trunc"] = trunc
#                 new_config["experiment"]["name"] = f"{method}_trunc{trunc}_iter{iter}"
#                 with open(
#                     f"/home/aosimul/noah/src/pipeline_ADI/configs/{new_config['experiment']['name']}.yaml",
#                     "w",
#                 ) as f:
#                     yaml.safe_dump(new_config, f, sort_keys=False)
#     else:
#         new_config["experiment"]["name"] = f"{method}"
#         with open(
#             f"/home/aosimul/noah/src/pipeline_ADI/configs/{new_config['experiment']['name']}.yaml",
#             "w",
#         ) as f:
#             yaml.safe_dump(new_config, f, sort_keys=False)

# run_pipeline(config)
folder = '/home/aosimul/noah/data/ghost_images/10_20ws/'

new_config["fake_planet"]["flux_ratio_mag"] = np.linspace(6.0, 14.0, 16)
new_config["fake_planet"]["components"] = np.concatenate(
        [
            np.arange(0,10, 1)[1:],
            np.arange(10, 50, 5),
            np.arange(50, 150, 15),
        ]
    )

for ro in [0.5,0.7,1.0]:
    for ws in [10, 20]:
        new_config["experiment"]["name"] = f"coro_{int(ro*10)}r0_{ws}windspeed"
        for name, ds in new_config["datasets"].items():
            ds["science_file"] = f"{folder}{name}_{ro}ro_{ws}ws.npy"
            ds["psf_file"] = f"{folder}psf_{name}_{ro}ro_{ws}ws.npy"


        run_pipeline(new_config)


curves_output_path = '/home/aosimul/noah/src/pipeline_ADI/results/contrast'
plot_all_best(curves_output_path)