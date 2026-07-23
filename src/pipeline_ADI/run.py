from pipeline_ADI.contrast_pipeline.post_processing import run_pipeline, load_config
import copy
import yaml

config = load_config(
    "/home/aosimul/noah/src/pipeline_ADI/configs/ghost.yaml",
    defaults_path="/home/aosimul/noah/src/pipeline_ADI/configs/default_values.yaml",
)

new_config = copy.deepcopy(config)


for method in ["gram", "lowrank"]:
    new_config["fake_planet"]["pca_method"] = method
    if method == "lowrank":
        for iter in [1, 2]:
            new_config["fake_planet"]["niter"] = iter
            for trunc in [10, 50, None]:
                new_config["fake_planet"]["approx_svd_trunc"] = trunc
                new_config["experiment"]["name"] = f"{method}_trunc{trunc}_iter{iter}"
                with open(
                    f"/home/aosimul/noah/src/pipeline_ADI/configs/{new_config['experiment']['name']}.yaml",
                    "w",
                ) as f:
                    yaml.safe_dump(new_config, f, sort_keys=False)
    else:
        new_config["experiment"]["name"] = f"{method}"
        with open(
            f"/home/aosimul/noah/src/pipeline_ADI/configs/{new_config['experiment']['name']}.yaml",
            "w",
        ) as f:
            yaml.safe_dump(new_config, f, sort_keys=False)

# run_pipeline(config)
