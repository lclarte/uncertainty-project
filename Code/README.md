### Organisation of the code

variances_gamp_erm.ipynb : main script, compares the different variances for Bayes-Optimal and ERM

### Running the experiments

The code use the package `sacred` to manage the experiments. This allows to customize the arguments in the CLI. For example, we can do
```
python -m experiments.check_erm with d=200 task='ridge'
```
All the experimental results are saved in the folder experiments_results + the appropriate folder (/check_erm or /check_bayes_optimal). They are stored in the form of .json files : 
- info.json contains the curves (losses, MSE)
- config.json contains the arguments used for the experiment (dimension, Ridge or Logistic, etc.) 
