1. How to run

> Executed on WSL2 of Ubuntu 22.04, python 3.10 with libraries present in requirements.txt


Commands to run the code:

- python3 -m venv venv/
- source venv/bin/activate
- pip install -r requirements.txt
- python3 independent_runs.py # Execute models and save results in json files.
- python3 plot_results.py # Plot and calculate confidence ranges from json files.

If preferred, the libraries have been manually installed with:

- pip install tensorflow[and-cuda]
- pip install numpy gaft matplotlib scikit-learn pandas cma cma-es
