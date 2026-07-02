import argparse

from .pipeline import (
    load_config,
    run_pipeline
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML config"
    )

    args = parser.parse_args()

    config = load_config(args.config, defaults_path="configs/default_values.yaml")
    run_pipeline(config)


if __name__ == "__main__":
    main()