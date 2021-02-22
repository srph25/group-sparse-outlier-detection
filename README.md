# Robust Detection of Anomalies via Sparse Methods
## Python source code for reproducing the experiments described in the paper
[Paper](https://argmax.ai/pdfs/brml/MilLudLorSma2015.pdf)\
Code is mostly self-explanatory via file, variable and function names; but more complex lines are commented.\
Designed to require minimal setup overhead.\
Note: as the Baxter Robot Arm data set from the paper is closed source, I am releasing very similar artificially generated data here.

### Installing dependencies
**Installing Python 3.7.9 on Ubuntu 20.04.2 LTS:**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```
**Installing Python packages with pip:**
```bash
python3.7 -m pip install cvxpy==1.1.10 ipython==7.16.1 joblib==1.0.1 matplotlib==3.3.2 numpy==1.19.2 scikit-learn==0.23.2 scipy==1.5.2 scs==2.1.2
```

### Running the code
Reproduction should be as easy as executing this in the root folder (after installing all dependencies):
```bash
python3.7 -m IPython run_toy.py
```


### Directory and file structure:
results/ : experimental results will be saved to this directory with numpy\
run_toy.py : conduct experiment on the artificially generated piecewise linear toy data set\
outlier_gflasso: our group sparsity based outlier detection method implemented in CVXPY.


### Contact:
In case of any questions, feel free to create an issue here on GitHub, or mail me at [srph25@gmail.com](mailto:srph25@gmail.com).

