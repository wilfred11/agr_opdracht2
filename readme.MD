This project was created while tackling an assignment on bayesian learnning in pyagrum, a Python package that can be used to infer, create, explore, compare, ... Bayesian Networks and derivatives.
I can only refer to a [pdf](https://github.com/wilfred11/agr_opdracht2/blob/master/opdracht2-brml-wvc.pdf) describing the different functions that were created to complete the assignment. This project only slightly differs from the version I used to write the pdf.

Instructions to use this project:
- Install pycharm (community edition)
- Open PyCharm
- From File menu select Project from version control
- In the terminal type poetry install (this command should install all required packages, except some globally required)
- Download and instal  graphviz for windows 12.2.1 / add  bin directory to path
- Download and install c++ build tools  14.0 (if not already installed)
- Reopen Pycharm, clean caches 
- Maybe even restart Windows 10/11

For my setup this was enough but this is not always the case

The packages needed for this setup are (to be found in [pyproject.toml](https://github.com/wilfred11/agr_opdracht2/blob/master/pyproject.toml))

python = "^3.11"

pyagrum = "^2.1.1"

pandas = "^2.3.1"

ipython = "^9.4.0"

scikit-learn = "^1.7.0"

tqdm = "^4.67.1"
