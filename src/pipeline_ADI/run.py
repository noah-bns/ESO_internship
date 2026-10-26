from pipeline_ADI.contrast_pipeline.post_processing import run_pipeline, load_config
import copy
import yaml, json
import numpy as np
from pipeline_ADI.contrast_pipeline.plotting import plot_all_best

config = load_config(
    "/home/aosimul/noah/src/pipeline_ADI/config_files/ghost.yaml",
    # defaults_path="/home/aosimul/noah/src/pipeline_ADI/config_files/default_values.yaml",
)

new_config = copy.deepcopy(config)

# %%--------------------------

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

# %%--------------------------
ori_folder = "/home/aosimul/noah/data/ghost_images/long_7ro_20ws_focus/"
new_config["experiment"]["ori_folder"] = ori_folder

new_config["fake_planet"]["flux_ratio_mag"] = 15  # np.linspace(4.5, 15.0, 30)
new_config["fake_planet"]["num_fake_planets"] = 6  # np.linspace(4.5, 15.0, 30)
compo = np.arange(1,100)
new_config["fake_planet"]["components"] = compo
# %%--------------------------
# for ro in [0.5,0.7,1.0]:
#     for ws in [10, 20]:
#         new_config["experiment"]["name"] = f"coro_{int(ro*10)}r0_{ws}windspeed"
#         for name, ds in new_config["datasets"].items():
#             ds["science_file"] = f"{folder}{name}_{ro}ro_{ws}ws.npy"
#             ds["psf_file"] = f"{folder}psf_{name}_{ro}ro_{ws}ws.npy"

# %%--------------------------

new_config["experiment"]["name"] = f"30_deg_sat_speckles"
new_config["instrument"]["radius_psf"] = 10
new_config["instrument"]["radius_sc"] = 45
new_config["fake_planet"]["max_separation"] = 1
# new_config["instrument"]["dit_psf"] = 1
new_config["datasets"]["int"]["enabled"] = True

# new_config["datasets"]["int"]["psf_factor"] = 1.416e3 * 1.219e6
# new_config["datasets"]["pred"]["psf_factor"] = 1.416e3 * 1.449e6
new_config["datasets"]["int"]["psf_file"] = "master_psf.npy"
new_config["datasets"]["pred"]["psf_file"] = "master_psf.npy"
new_config["contrast"]["range_constrast_y"] = [16, 4]


DIT_sci = 2.814e-3
new_config["instrument"]["dit_science"] = DIT_sci
new_config["instrument"]["dit_psf"] = None
new_config["instrument"]["science_frames"] = None
files_int = []
files_pred = []
name = "_7ro_20ws_"
for key in [1, 24, 32]:  #
    files_int.append(f"int{name}{key}key_1-4.npy")
    files_pred.append(f"pred{name}{key}key_1-4.npy")

new_config["datasets"]["int"]["science_file"] = files_int
new_config["datasets"]["pred"]["science_file"] = files_pred
new_config["datasets"]["int"]["dit_psf"] = 1 / (1.416e3 * 8.611e5)
new_config["datasets"]["pred"]["dit_psf"] = 1 / (1.416e3 * 1.019e6)
new_config["fake_planet"]["visualisation"] = True

for dit_factor in [10]:
    # new_config["instrument"]["science_frames"] = 50000
    new_config["instrument"]["dit_factor"] = dit_factor
    new_config["experiment"]["extrasubname"] = f"DIT_{dit_factor*DIT_sci:.2e}s"
    run_pipeline(new_config)


# for count in [2]:
#     for ro in [10, 5, 7]:
#         for ws in [10]:
#             name = f"{ro}ro_{ws}ws"
#             with open("/home/aosimul/noah/notebooks/light_intensities.json", "r") as f:
#                 light_intensities = json.load(f)
#                 sat_int = light_intensities[f'int_{name}_{count}']['satellite']
#                 sat_pred = light_intensities[f'pred_{name}_{count}']['satellite']
#             new_config["datasets"]["int"]["science_file"] = f"int_{name}_{count}.npy"
#             new_config["datasets"]["pred"]["science_file"] = f"pred_{name}_{count}.npy"
#             new_config["datasets"]["int"]["psf_factor"] = 1.416e3 * sat_int
#             new_config["datasets"]["pred"]["psf_factor"] = 1.416e3 * sat_pred
#             run_pipeline(new_config)

# with open(
#     f"/home/aosimul/noah/src/pipeline_ADI/configs/{new_config['experiment']['name']}.yaml",
#     "w",
# ) as f:
#     yaml.safe_dump(new_config, f, sort_keys=False)


curves_output_path = "/home/aosimul/noah/src/pipeline_ADI/results/contrast"
plot_all_best(curves_output_path)

# %%--------------------------
