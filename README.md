# FiLM-SimVP: Scalable Uncertainty Quantification in Spatiotemporal Forecasting

This repository contains the official implementation of FiLM-SimVP, which uses Feature-wise Linear Modulation (FiLM) for quantile regression in spatiotemporal forecasting tasks.

## Overview

FiLM-SimVP extends SimVP with uncertainty quantification capabilities by using feature-wise affine transformations to predict multiple quantiles simultaneously. The model provides probabilistic forecasts by outputting prediction intervals rather than just point estimates.

## Installation

```sh
# Clone the repository
git clone https://github.com/username/FiLM-SimVP.git
cd FiLM-SimVP

# Create conda environment
conda env create -f environment.yml

# Install requirements
pip install -r requirements.txt
```

## Usage

### Training

To train the model, use the training script:

```sh
bash tools/dist_train.sh
```

### Testing

For evaluation:

```sh 
bash tools/dist_test.sh
```

## Key Components

- `PinballLoss`: Implementation of the quantile loss function
- `WeightedMISLoss`: Multiple Interval Score loss for interval predictions
- `TaxibjDataset`: Dataset loader with quantile sampling

## Datasets

The code supports multiple datasets including:
- TaxiBJ
- Traffic4Cast
- WeatherBench

Dataset configurations can be found in the configs directory.

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{
  title={FiLM-SimVP: Scalable Uncertainty Quantification in Spatiotemporal Forecasting},
  author={},
  journal={},
  year={}
}
```

## License

GPL-3.0 license

## Acknowledgements

This work builds upon OpenSTL for spatiotemporal axirning.
