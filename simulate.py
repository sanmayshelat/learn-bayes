import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from dgp import DgpProportion, ExperimentImpact
import warnings
warnings.filterwarnings("ignore")


def calc_ptb_loss(posterior_c, posterior_t):
    ptb = (posterior_t > posterior_c).mean()
    loss = posterior_t - posterior_c
    loss[loss > 0] = np.nan
    exp_loss = np.nanmean(loss)
    return ptb, exp_loss

def calc_ptb(posterior_c, posterior_t):
    ptb = (posterior_t > posterior_c).mean()
    return ptb

def calc_loss(posterior_c, posterior_t):
    loss = posterior_t - posterior_c
    loss[loss > 0] = np.nan
    exp_loss = np.nanmean(loss)
    return exp_loss

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

def simulate(
        n_units=20_000, 
        prob_success=0.02, #TODO: maybe change to mean_prob_success?
        eff_sample_size=None,
        prior_alphas=[1,1],
        prior_betas=[1,1], # uninformative prior
        independent=False,
        rel_lift_prior_mean = [0],
        rel_lift_prior_std=[100],  # uninformative prior
        perc_change=[0, 10], 
        samples=100, 
        seed=None
):

    # Setup experiment and data generation
    expt = ExperimentImpact(perc_change=perc_change)
    dgp = DgpProportion(prob_success=prob_success, eff_sample_size=eff_sample_size)
    stream_df = dgp.dgp(n_units=n_units, expt_impact=expt)

    # Seeds for Bayesian sampling
    if seed is None:
        seed = np.random.default_rng().integers(0, 1000)
    rng = [np.random.default_rng(seed=seed+i) for i in range(len(prior_alphas))] # as many rngs as experiment groups
    control_rng = rng[0]
    treatment_rng = rng[1] #TODO: Generalize to accept multiple treatments
    uplift_rng = np.random.default_rng(seed=seed+54)

    # Summarize outcomes
    ## Control
    stream_df['control'] = stream_df['treatment']==0
    stream_df['control_successes_cum'] = (stream_df['outcome'] * stream_df['control']).cumsum()
    stream_df['control_cum'] = stream_df['control'].cumsum()
    stream_df['control_rate'] = (stream_df['control_successes_cum'] + 0.5) / (stream_df['control_cum'] + 1)
    
    ## Treatment
    stream_df['treatment_successes_cum'] = (stream_df['outcome'] * stream_df['treatment']).cumsum()
    stream_df['treatment_cum'] = stream_df['treatment'].cumsum()
    stream_df['treatment_rate'] = (stream_df['treatment_successes_cum'] + 0.5) / (stream_df['treatment_cum'] + 1)
    
    ## Lift
    stream_df['rel_lift'] = stream_df['treatment_rate'] / stream_df['control_rate'] - 1
    # Apply the delta method to get the variance of the ratio
    stream_df['control_rate_var'] = stream_df['control_rate'] * (1 - stream_df['control_rate']) / (stream_df['control_cum'])
    stream_df['treatment_rate_var'] = stream_df['treatment_rate'] * (1 - stream_df['treatment_rate']) / (stream_df['treatment_cum'])
    delta_method_term_1 = stream_df['treatment_rate_var'] / (stream_df['control_rate']**2)
    delta_method_term_2 = 0 # because covariance is 0 in an A/B test
    delta_method_term_3 = stream_df['control_rate_var'] * (stream_df['treatment_rate']**2) / (stream_df['control_rate']**4)
    stream_df['rel_lift_var'] = delta_method_term_1 + delta_method_term_2 + delta_method_term_3
    stream_df['rel_lift_precision'] = 1 / stream_df['rel_lift_var']
    stream_df['num_samples'] = stream_df['control_cum'] + stream_df['treatment_cum']


    # Beta priors and update based on outcomes
    control_alpha_prior, control_beta_prior= prior_alphas[0], prior_betas[0]
    stream_df['control_alpha_posterior'] = control_alpha_prior + stream_df['control_successes_cum']
    stream_df['control_beta_posterior'] = control_beta_prior + stream_df['control_cum'] - stream_df['control_successes_cum']
    stream_df['control_posterior_samples'] = stream_df.apply(lambda x: 
        control_rng.beta(a=x.control_alpha_posterior, b=x.control_beta_posterior, size=samples), axis=1
    )
    
    if independent:
        treatment_alpha_prior, treatment_beta_prior = prior_alphas[1], prior_betas[1] #TODO: Generalize to accept multiple treatments
        stream_df['treatment_alpha_posterior'] = treatment_alpha_prior + stream_df['treatment_successes_cum']
        stream_df['treatment_beta_posterior'] = treatment_beta_prior + stream_df['treatment_cum'] - stream_df['treatment_successes_cum']
        #TODO: Check
        stream_df['treatment_posterior_samples'] = stream_df.apply(lambda x: 
            treatment_rng.beta(a=x.treatment_alpha_posterior, b=x.treatment_beta_posterior, size=samples), axis=1
        ) #TODO: Generalize to accept multiple treatments
        stream_df['rel_uplift_posterior_samples'] = stream_df.apply(lambda x:
            x.treatment_posterior_samples / x.control_posterior_samples - 1, axis=1
        )
    else:
        # If correlated, normal prior for uplift and no prior for treatment
        # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
        rel_lift_prior_var = rel_lift_prior_std[0] ** 2
        stream_df['rel_lift_posterior_var'] = (
            (1/rel_lift_prior_var) + (1/stream_df['rel_lift_var'])
        ) ** (-1)
        stream_df['rel_lift_posterior_mean'] = (
            rel_lift_prior_var * stream_df['rel_lift']/(rel_lift_prior_var + stream_df['rel_lift_var'])
            + stream_df['rel_lift_var'] * rel_lift_prior_mean[0] / (rel_lift_prior_var + stream_df['rel_lift_var'])
        )
        #TODO: Check
        stream_df['rel_uplift_posterior_samples'] = stream_df.apply(lambda x:
            uplift_rng.normal(loc=x.rel_lift_posterior_mean, scale=np.sqrt(x.rel_lift_posterior_var), size=samples), axis=1
        ) #TODO: Generalize to accept multiple treatments
        stream_df['treatment_posterior_samples'] = stream_df.apply(lambda x:
            x.control_posterior_samples * (1 + x.rel_uplift_posterior_samples), axis=1
        )

    # Calculate PTB and expected loss
    stream_df['ptb'] = stream_df.apply(lambda x: calc_ptb(x.control_posterior_samples, x.treatment_posterior_samples), axis=1)
    stream_df['exp_loss'] = stream_df.apply(lambda x: calc_loss(x.control_posterior_samples, x.treatment_posterior_samples), axis=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in %")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero%")

        stream_df['pvalue'] = stream_df.apply(
        lambda x: proportions_ztest(
            count=[x.control_successes_cum, x.treatment_successes_cum], 
            nobs=[x.control_cum, x.treatment_cum]
        )[1], axis=1)

    return stream_df



# multiprocessing worker
def run_simulation(_):
    return simulate(
        independent=False,
)

def run_simulation_ind(_):
    return simulate(
        independent=True,
    )

def run_simulation_ind_informative(_):
    # Using 30/970 parameters for control and treatment to give mean ~0.03 with 95% mass between 0.01-0.05
    return simulate(
        independent=True,
        prior_alphas=[30, 30],
        prior_betas=[970, 970],
    )

def run_simulation_informative(_):
    # Using 30/970 parameters for control to give mean ~0.03 with 95% mass between 0.01-0.05
    # And setting the relative lift prior with mean 0 and std 0.05 (95% mass between -0.1 and +0.1)
    return simulate(
        independent=False,
        prior_alphas=[30, 30],
        prior_betas=[970, 970],
        rel_lift_prior_mean=[0],
        rel_lift_prior_std=[0.1],
    )