This repository has been tested on macOS Sonoma 14.4.1.
Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.12" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.
    * Within the conda environment it's possible to install the requirements with pip.

## Repositories
* GitHub Actions has been set up, more details can be found in ```.github/workflows/main.yml```

# Data
* The raw and cleaned version of the dataset can be found in ```./data```

# Model
* In order to train from scratch the ML model, run ```python train_model.py```
  * An updated version of model, encoder and label binarizer will be stored as .pkl files in ```./model```
* All unit tests can be found in ```./tests```
* To evaluate the performance of the model on different slices of data, run ```python compute_slice_metrics.py``` after changing the ```slice_feature``` variable.
* The model card is described in model_card.md

# API Creation
*  A RESTful API using FastAPI, including a get and post methods, is implemented in 

# API Deployment
* As this guide is being written, a live API is available [here](https://im-tired-boss2-9c09a0b9b543.herokuapp.com/predict).
* The ```live_post.py``` script allows to test both methods of the deployed app.
