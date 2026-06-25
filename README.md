# GHOST High-Contrast Imaging Pipeline

ESO Internship project for developing and testing high-contrast imaging techniques on the GHOST spectrograph.

## Overview

This project implements a comprehensive pipeline for processing and analyzing high-contrast imaging data from astronomical observations. The pipeline includes:

- Data preprocessing and calibration
- High-contrast imaging algorithms
- Analysis and visualization tools
- Configuration management for different observational setups

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/noah-bns/ESO_internship.git
cd ESO_internship
```

2. Install dependencies:
```bash
pip install -r science_requirements.txt
```

3. Configure the pipeline:
```bash
cp config/default_config.yaml config/my_config.yaml
# Edit config/my_config.yaml with your settings
```

## Project Structure

```
ESO_internship/
├── src/                      # Main source code
│   ├── preprocessing.py      # Data calibration and preprocessing
│   ├── imaging.py            # High-contrast imaging algorithms
│   ├── analysis.py           # Data analysis tools
│   └── utils.py              # Utility functions
├── config/                   # Configuration files
│   └── default_config.yaml   # Default pipeline configuration
├── notebooks/                # Jupyter notebooks for analysis
├── science_requirements.txt   # Project dependencies
└── README.md                 # This file
```

## Usage

### Basic Pipeline Run

```python
from src.preprocessing import preprocess_data
from src.imaging import apply_high_contrast_imaging

# Load and preprocess data
data = preprocess_data('raw_data.fits')

# Apply high-contrast imaging
processed = apply_high_contrast_imaging(data, method='contrast_enhancement')

# Analyze results
results = analyze_data(processed)
```

### Using Configuration Files

```python
import yaml
from src.pipeline import Pipeline

with open('config/my_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

pipeline = Pipeline(config)
results = pipeline.run('observation_data.fits')
```

## Configuration

Edit `config/default_config.yaml` to customize:

- Data paths and file formats
- Imaging algorithm parameters
- Processing options
- Output settings

## Development

To contribute to this project:

1. Create a new branch for your feature
2. Make your changes and test thoroughly
3. Submit a pull request with a description of changes

## Requirements

See `science_requirements.txt` for the complete list of dependencies including:

- numpy
- scipy
- astropy
- matplotlib
- pyyaml

## References

- GHOST Spectrograph Documentation
- High-Contrast Imaging Techniques
- ESO Data Processing Guidelines

## License

[Add license information here]

## Contact

For questions or issues, please open an issue on this repository.
