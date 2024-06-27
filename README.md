# TFG Wrapper-Based Metaheuristics Feature Selection

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Acknowledgments](#acknowledgments)

## Introduction

This repository contains the code and resources for the Final Year Project titled "Wrapper-Based Metaheuristics for Feature Selection." The project aims to explore and analyze modern metaheuristic algorithms applied to feature selection in machine learning.

In Machine Learning, algorithms like **kNN** and **SVM** show excellent results but often require preprocessing to identify relevant features. This preprocessing, known as feature selection, is a complex combinatorial optimization problem. Common methods include filtering, wrapping, tree-based methods, PCA, and L1 selection.

Metaheuristics, designed to solve complex optimization problems with limited resources, have been adapted for feature selection. This work reviews recent metaheuristics, implements the most promising ones, and conducts a comprehensive comparative study using various ML algorithms and datasets, evaluating metrics such as accuracy and execution time.


## Features

- Implementation of various metaheuristic algorithms in its binary version.
- Wrapper-based feature selection approach.
- Extensive documentation and analysis.
- Includes scripts for running experiments, generating results and plots.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Migue8gl/TFG-Wrapper-Based-Metaheuristics-Feature-Selection.git
cd TFG-Wrapper-Based-Metaheuristics-Feature-Selection
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

## Usage

### Running the Main Script

Run the main script for optimizing a problem with the following command:

```bash
python src/main.py \
    -d DATASET \
    -i EPOCHS \
    -o OPTIMIZER \
    -v VERBOSE \
    -b ENCODING \
    -s SCALING

# Command-line arguments:
# -d DATASET: Specify the dataset to use for training and evaluation
# -i EPOCHS: Set the number of training epochs
# -o OPTIMIZER: Choose the optimization algorithm (e.g., GWO, PSO)
# -v VERBOSE: Set the verbosity level for output (0 for quiet, 1 for activated)
# -b ENCODING: Specify the encoding method for categorical variables (binary with s-shaped <s>, binary with v-shaped <v>, real encoding <r>)
# -s SCALING: Choose the type of normalization or scaling for numerical features

## Directory Structure
TFG-Wrapper-Based-Metaheuristics-Feature-Selection/
│
├── data/                     # Data files for the experiments
├── docs/                     # Documentation files
├── scripts/                  # Scripts for running experiments
├── slides/                   # Presentation slides
├── src/                      # Source code of the project
├── utils/                    # Some tools
├── LICENSE                   # License file
├── README.md                 # README file
├── requirements.txt          # Python dependencies
└── MiguelGarcíaLópez-EstudioYAnalisisDeMetaheuristicasModernasFeatureSelection.pdf  # Project report

## Acknowledgments

We would like to express our sincere gratitude to the contributors of the [pyMetaheuristic](https://github.com/Valdecy/pyMetaheuristic) repository. Their work provided the original code for the optimization algorithms used in this project. Their efforts in developing and sharing these metaheuristic algorithms have been instrumental in the development of our project.

Special thanks to:
- [Valdecy Pereira](https://github.com/Valdecy), the main contributor of pyMetaheuristic
- All other contributors to the pyMetaheuristic project

Their open-source contribution has been invaluable in advancing the field of optimization and enabling further research and development in this area.

