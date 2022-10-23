from model import Model
import numpy as np
from numpy.random import SeedSequence
from joblib import Parallel, delayed
from itertools import product

def kwparams(C, beta, trial):
    params = {
        "maxsteps" : 1000000,
        "tolerance" : 0.00001,
        "alpha" : 0.1,
        "C" : C,
        "tolerance_upper" : 0.4,
        "tolerance_lower" : 0.1,
        "C" : C,
        "beta" : beta,
        "synthetic" : 'yes',
        "dataset" : 'Reddit',
        "N" : 1000,
        "p" : 0.01,
        "M" : 1,
        "K" : 5,
        "trial" : trial,
        "fulltimeseries" : False
    }
    return params

def run_model(seedseq, model_params):
    model = Model(seedseq, **model_params)
    model.run()


if __name__ == '__main__':

    workers = 100
    seed = 1615946511
    #seed = 1613961486

    beta_range = list(np.round(np.arange(0.1,1.02,0.02),3))
    #C_range = list(np.round(np.arange(0.1,0.402,0.02),3))
    C_range = [1]
    # trial_range = list(range(1,51))
    trial_range = list(range(1,21))
    num_runs = len(C_range) * len(beta_range) * len(trial_range)
    param_grid = product(C_range,beta_range,trial_range)

    ssq = SeedSequence(seed)
    childstates = ssq.spawn(num_runs)

    with Parallel(n_jobs=workers, verbose=1) as parallel:
        parallel(delayed(run_model)(s, kwparams(*p)) for s,p in zip(childstates, param_grid))
