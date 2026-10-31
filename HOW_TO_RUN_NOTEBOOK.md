# Jupyter Notebook Guide

The notebook is toolmatcher_ai_advanced.ipynb. Open it with RUN_NOTEBOOK.bat, Jupyter Notebook, JupyterLab, or VS Code.

## What the notebook contains

It records an earlier exploratory training run, including data inspection, preprocessing, model comparisons, and charts. Its saved outputs report a 3,360-row cleaned dataset and a 97.02% stacking-ensemble result. Those outputs are historical and do not describe the model currently included with the Flask app.

For the current repository snapshot, the cleaned dataset has 339 rows. The committed Random Forest has a recorded 76.47% accuracy on its 68-row holdout. See PROJECT_REPORT.md and ML_TRAINING_REPORT.md for the current analysis.

## Open the notebook

On Windows, double-click RUN_NOTEBOOK.bat. Or from the project directory, run:

    jupyter notebook toolmatcher_ai_advanced.ipynb

To use JupyterLab:

    jupyter lab toolmatcher_ai_advanced.ipynb

Jupyter is included in requirements.txt.

## Rebuild the current model

The supported current training workflow is in the Python scripts:

    python data_cleaning_pipeline.py
    python quick_model_training.py

The advanced comparison is available separately:

    python advanced_model_training.py

The notebook contains saved outputs from an earlier experiment. Rerunning its cells that save models or data can replace files used by the app, so use the Python scripts above when you intend to rebuild the current workflow.
