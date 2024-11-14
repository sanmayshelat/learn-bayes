import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from dgp import DgpProportion, ExperimentImpact
import warnings
warnings.filterwarnings("ignore")

# import polars as pl

def calc_ptb(posterior_c, posterior_t):
    return (posterior_t > posterior_c).mean()

# def calc_hdi(arr: np.array, hdi_prob: float = 0.95):
#     arr = arr.flatten()
#     n = len(arr)
#     arr = np.sort(arr)
#     interval_idx_inc = int(np.floor(hdi_prob * n))
#     n_intervals = n - interval_idx_inc
#     interval_width = arr[interval_idx_inc:] - arr[:n_intervals]
#     min_idx = np.argmin(interval_width)
#     hdi_min = arr[min_idx]
#     hdi_max = arr[min_idx + interval_idx_inc]
#     hdi_interval = np.array([hdi_min, hdi_max])
#     return hdi_interval

def simulate(n_units=100_000, perc_change=[0, 0], samples=1000, seed=None):
    expt = ExperimentImpact(perc_change=perc_change)
    dgp = DgpProportion(prob_success=0.02)
    stream_df = dgp.dgp(n_units=n_units, expt_impact=expt)

    if seed is None:
        seed = np.random.default_rng().integers(0, 1000)
    control_rng = np.random.default_rng(seed=seed)
    treatment_rng = np.random.default_rng(seed=seed+27)

    control_alpha_prior = 1
    control_beta_prior = 1
    treatment_alpha_prior = 1
    treatment_beta_prior = 1

    stream_df['control'] = stream_df['treatment']==0
    stream_df['control_successes_cum'] = (stream_df['outcome'] * stream_df['control']).cumsum()
    stream_df['treatment_successes_cum'] = (stream_df['outcome'] * stream_df['treatment']).cumsum()
    stream_df['control_cum'] = stream_df['control'].cumsum()
    stream_df['treatment_cum'] = stream_df['treatment'].cumsum()

    stream_df['control_alpha_posterior'] = control_alpha_prior + stream_df['control_successes_cum']
    stream_df['control_beta_posterior'] = control_beta_prior + stream_df['control_cum'] - stream_df['control_successes_cum']

    stream_df['treatment_alpha_posterior'] = treatment_alpha_prior + stream_df['treatment_successes_cum']
    stream_df['treatment_beta_posterior'] = treatment_beta_prior + stream_df['treatment_cum'] - stream_df['treatment_successes_cum']

    stream_df['ptb'] = stream_df.apply(lambda x: calc_ptb(
        control_rng.beta(a=x.control_alpha_posterior, b=x.control_beta_posterior, size=samples),
        treatment_rng.beta(a=x.treatment_alpha_posterior, b=x.treatment_beta_posterior, size=samples) 
    ), axis=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in %")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero%")

        stream_df['pvalue'] = stream_df.apply(
        lambda x: proportions_ztest(
            count=[x.control_successes_cum, x.treatment_successes_cum], 
            nobs=[x.control_cum, x.treatment_cum]
        )[1], axis=1)

    return stream_df

# Attempt with Polar
# expt = ExperimentImpact(perc_change=[0,0])
# dgp = DgpProportion(prob_success=0.02)
# stream_df = dgp.dgp(n_units=100_000, expt_impact=expt)
# stream_df = pl.from_pandas(stream_df)
# seed = None
# if seed is None:
#     seed = np.random.default_rng().integers(0, 1000)
# control_rng = np.random.default_rng(seed=seed)
# treatment_rng = np.random.default_rng(seed=seed+27)
# control_alpha_prior = 1
# control_beta_prior = 1
# treatment_alpha_prior = 1
# treatment_beta_prior = 1
# (
#     stream_df
#         .with_columns((pl.col('treatment')==0).alias('control'))
#         .with_columns([
#             ((pl.col('outcome') * pl.col('control')).cum_sum()).alias('control_successes_cum'),
#             ((pl.col('outcome') * pl.col('treatment')).cum_sum()).alias('treatment_successes_cum'),
#             ((pl.col('control')).cum_sum()).alias('control_cum'),
#             ((pl.col('treatment')).cum_sum()).alias('treatment_cum'),
#         ])
#         .with_columns([
#             (control_alpha_prior + pl.col('control_successes_cum')).alias('control_alpha_posterior'),
#             (control_beta_prior + pl.col('control_cum') - pl.col('control_successes_cum')).alias('control_beta_posterior'),
#             (treatment_alpha_prior + pl.col('treatment_successes_cum')).alias('treatment_alpha_posterior'),
#             (treatment_beta_prior + pl.col('treatment_cum') - pl.col('treatment_successes_cum')).alias('treatment_beta_posterior'),
#         ])
# )

# multiprocessing worker
def run_simulation(_):
    return simulate()