# Prediction of Metal Nanoparticle Energies using Machine Learning

This repository contains code and data for predicting the total and formation energies of metal nanoparticles (specifically gold, palladium, and platinum, and their combinations) using machine learning models. The goal is to explore the feasibility of using most archetypal samples as a representative of the whole set of training data.

## Project Overview

The project involves a full machine learning pipeline, including:
- **Data Processing:** Cleaning and preparing the nanoparticle datasets.
- **Feature Engineering & Selection:** Removing low-variance and highly correlated features to improve model performance and interpretability.
- **Model Training:** Training various regression models, including Random Forest and Linear Regression.
- **Evaluation:** Assessing model performance using metrics such as R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- **Visualization:** Generating learning curves, correlation heatmaps, and feature importance plots.

## Repository Structure

- `data/`: Contains raw CSV datasets (Au, Pd, Pt nanoparticles), processed data in pickle format, and hyperparameter tuning results.
- `figs/`: Stores generated visualizations, including correlation heatmaps, learning curves, and feature importance rankings.
- `trainEvalModels.py`: A Python script containing utility functions for data splitting, scaling, feature selection, and model evaluation.
- `metal-nanoparticle-energy-regression-data-processing-data.ipynb`: Jupyter notebook detailing the data processing and reduction steps.
- `metal-nanoparticle-energy-regression-model-train-eval.ipynb`: Jupyter notebook covering the model training and evaluation process.
- `requirements.txt`: List of Python dependencies required to run the project.

## Installation

To set up the environment, it is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

1.  **Data Processing:** Follow the steps in `metal-nanoparticle-energy-regression-data-processing-data.ipynb` to prepare the datasets.
2.  **Model Training and Evaluation:** Use `metal-nanoparticle-energy-regression-model-train-eval.ipynb` to train the models and analyze their performance.
3.  **Utility Functions:** The `trainEvalModels.py` script can be imported into other scripts or notebooks for reusable machine learning tasks.

## License

This project is licensed under the terms provided in the `LICENSE` file.
