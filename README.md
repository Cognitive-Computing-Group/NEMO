# NEMO

This is the accompanying code repository for the NEMO dataset. The fNIRS data can be found at https://osf.io/pd9rv/.

To get started using the dataset, see the [Usage](#usage) section.

To reproduce the results from the paper, see the [Automatic workflow](#automatic-workflow) section.

To export the data to other formats, see the [Exporting data](#exporting-data) section.

# Usage

## Data

The data is available in two formats, BIDS and CSV.

#### 1. BIDS

BIDS contains the raw optical density (OD) recordings and corresponding metadata for each participant. This code repository uses the BIDS data to produce the results in the paper. Start using the BIDS data by downloading [nemo-bids](https://osf.io/tcak6) and placing it in the `data/` folder.

#### 2. CSV

The CSV format provides an easy way to access the processed epochs data without needing any code from this repository or other BIDS tools. The CSV data does not offer anything additional to the BIDS data, but it might be more convenient for, e.g. machine learning tasks. To use the CSV data, download [empe_csv](https://osf.io/m4kgn) (Emotional perception) or [afim_csv](https://osf.io/xq2v6) (Affective imagery) and place it in, e.g. the `data/` folder. [scripts/notebooks/classification_from_csv.ipynb](scripts/notebooks/classification_from_csv.ipynb) demonstrates how to use the CSV data for a simple classification task.

## Installation

via pip:
```bash
pip install -e . # run in the main directory
```

via conda:
```bash
conda env create -f environment.yaml -n nemo
conda activate nemo
```

## Quickstart

The easiest way to start using the dataset is by first loading the BIDS data to MNE format by running:

```bash
python scripts/load_bids.py
```

And then running the following interactive jupyter notebooks:

[scripts/notebooks/classification_simple.ipynb](scripts/notebooks/classification_simple.ipynb) demonstrates how to use the data for classification.

[scripts/notebooks/plotting.ipynb](scripts/notebooks/plotting.ipynb) shows how to plot the data.

#### Scripts

You can also run the classification pipeline from the command line:

```bash
# Load data
python scripts/load_bids.py
# Create the classification dataset
python scripts/create_dataset.py
# Run classification
python scripts/paper_results/run_clf.py
```

`scripts/paper_results/` contains other scripts for recreating the results from the paper. The scripts need to be run in a specific order, so we recommend using the [automatic workflow](#automatic-workflow) for reproducing the results.

## Automatic workflow

The results can be reproduced by running the workflow with [snakemake](https://snakemake.readthedocs.io/en/stable/). Snakemake is an alternative to Makefile that uses python syntax. The workflow is defined in `Snakefile`, and it uses the configuration file `config.yaml`.

Running the workflow requires the [BIDS](#1-bids) data to be downloaded and placed in the `data` folder.

```bash
conda create -c bioconda -c conda-forge -n snakemake-minimal snakemake-minimal -y # install snakemake
conda activate snakemake-minimal # activate the environment
snakemake -c --use-conda # run the workflow
```

Snakemake will run the scripts in a suitable order. The results will be saved in the `results/paper_results` folder. The folder structure is as follows:

```bash
paper_results # folder containing the results from the paper
├── tables # classification accuracies
├── sub_scores # subject-specific classification accuracy plots
├── response_plots # joint and topoplots for stimuli types
├── val_aro # ground truth valence and arousal plot
```

## Exporting data

`python scripts/load_bids.py` provides options for saving the  data as MNE objects at different stages of data processing. The options are:
1. Before preprocessing (raw OD, continuous) by including argument `--save_od`, e.g. `python scripts/load_bids.py --save_od`. The data will be saved in `processed_data/od/sub-<sub_id>_task-<task_id>_od.fif`.
2. After preprocessing (haemoglobin concentrations, continuous) by including argument `--save_haemo`. The data will be saved in `processed_data/haemo/sub-<sub_id>_task-<task_id>_haemo.fif`.
3. After epoching (haemoglobin concentrations, epochs). Done by default, the epochs are saved in `processed_data/epochs/sub-<sub_id>_task-<task_id>_epo.fif`.

The MNE objects can be exported to other formats if needed, see [exporting data from MNE](https://mne.tools/stable/export.html).

# Configuration

You can configure data directory paths and other parameters in the `config.yaml` file. If a path starts with `./`, it is interpreted as relative to the main directory. Otherwise, it is interpreted as absolute. By default, they are:

```
bids_path: './data/nemo-bids'
epochs_path: './processed_data/epochs'
```

Other config parameters control default values.

# Documentation

The code is documented with NumPy-style docstrings.

# Troubleshooting

### Jupyter notebook

You might get an internal server error because of the following:

```
PermissionError: [Errno 13] Permission denied: '/usr/local/share/jupyter/nbconvert/templates/latex/conf.json'
```

This is a known jupyter issue discussed here:
https://github.com/jupyter/nbconvert/issues/1594

One solution is just to give all permissions to the folder:

`sudo chmod -R 777 /usr/local/share/jupyter/nbconvert/templates/`
