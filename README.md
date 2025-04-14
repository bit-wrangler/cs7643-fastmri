# CS7643 FastMRI Project

This project works with the fastMRI dataset for MRI reconstruction tasks. The repository contains a Jupyter notebook for exploring and processing the fastMRI knee singlecoil dataset.

## Getting Started

Follow these instructions to set up your development environment and get the project running locally.

### Prerequisites

- Python 3.10
- Conda or Miniconda (recommended for environment management)
- fastMRI dataset (can be downloaded from https://fastmri.med.nyu.edu/)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cs7643-fastmri
   ```

2. Create and activate the conda environment using the provided environment.yml file:
   ```bash
   conda env create -f environment.yml
   conda activate cs7643-fastmri
   ```

### Environment Variables

1. Create a copy of the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file and update the paths to match your local setup:
   ```
   SINGLECOIL_TRAIN_PATH=/path/to/your/singlecoil_train
   SINGLECOIL_VAL_PATH=/path/to/your/singlecoil_val
   ```

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the `fastMRI_knee_singlecoil.ipynb` notebook

3. Run the cells to explore and process the fastMRI dataset
