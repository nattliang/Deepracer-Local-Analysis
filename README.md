# Deepracer Local Analysis
Performance visualization for DRfC local training

# How to use: 
Copy the .ipynb and .py files to the DRfC directory. Press the double arrow icon at the top to run while the training is running to load the training graphs.

# How it works:
The tool locates the TrainingMetrics logs based on the DR_LOCAL_S3_MODEL_PREFIX and the DR_LOCAL_S3_BUCKET in the run.env and system.env files. Then it loads the data and generates the training graphs.
