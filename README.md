# Echo_cancellation
------------
## Install

There is `requirements.txt`

`$ pip install -r requirements.txt`

## File

`utils.py` consists of load_data(), write_data(), both of function used soundfile library.

`main_2.py` is filled with NLMS(Normalized Least-Mean-Squares) filter

## Run

`$ python main_2.py`
If you need to change hyperparameters like mu and delta, you can change it in `main_2.py`

## About main_2.py

the final version of `main_2.py` is possible to save `.wav` file and graph of desired input and error signal and Ensemble Average Error.  

------------
##### All of codes and README.md is written for practice. It may have many errors.
