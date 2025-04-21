import numpy as np
from scipy.stats import norm, t

def grad_single_asset(theta, x, q=0.95, gamma=1e-8):
    """
    Calculate the stochastic gradient of the single asset objective function.
    """
    indicator = 1.0 if x > theta else 0.0
    return 1.0 - (1.0/(1.0-q))*indicator + 2.0*gamma*theta

def two_phase_sgld(distribution, params, q=0.95, gamma=1e-8,
                   beta=1e8,
                   lr1=1e-4, n1=10**6, burn1=10**3,
                   lr2=1e-5, n2=10**5, burn2=0):


    theta = 0.0
    sqrt1 = np.sqrt(2*lr1/beta)
    for i in range(n1):
        x = (np.random.normal(*params) if distribution=='normal'
             else t.rvs(params[0]))
        g = grad_single_asset(theta, x, q=q, gamma=gamma)
        theta -= lr1*g
        theta += sqrt1*np.random.randn()
    theta_star = theta


    thetas2, xs2 = [], []
    lr, sqrt2 = lr2, np.sqrt(2*lr2/beta)
    theta = theta_star
    for i in range(n2):
        x = (np.random.normal(*params) if distribution=='normal'
             else t.rvs(params[0]))
        g = grad_single_asset(theta, x, q=q, gamma=gamma)
        theta -= lr*g
        theta += sqrt2*np.random.randn()
        if i >= burn2:
            thetas2.append(theta)
            xs2.append(x)
    return np.array(thetas2), np.array(xs2)

def estimate_var_std(thetas):

    m = thetas.mean()
    s = thetas.std(ddof=1)
    return m, s

def estimate_cvar_and_std(theta_samples, x_samples, q=0.95):
    """
    On each pair (θ_i, x_i) compute
       V_i = θ_i + (1/(1-q)) * max(x_i - θ_i, 0)

    return mean±std for CVaRSGLD.
    """
    Vs = theta_samples + (1/(1-q)) * np.maximum(x_samples - theta_samples, 0)
    m = Vs.mean()
    s = Vs.std(ddof=1)
    return m, s

# ---- run experiments ----
q_values = [0.95, 0.99]
normal_params = [(0,1), (1,2), (3,5)]
t_dfs = [10,7,3]

print("Single asset (normal):")
for q in q_values:
    print(f"\nq = {q}")
    for params in normal_params:
        th2, xs2 = two_phase_sgld('normal', params, q=q)
        var_m, var_s = estimate_var_std(th2)
        cvar_m, cvar_s = estimate_cvar_and_std(th2, xs2, q=q)
        print(f" N{params}: VaR_SGLD = {var_m:.4f} (std {var_s:.4f}), "
              f"CVaR_SGLD = {cvar_m:.4f} (std {cvar_s:.4f})")

print("\nSingle asset (Student t):")
for q in q_values:
    print(f"\nq = {q}")
    for df in t_dfs:
        th2, xs2 = two_phase_sgld('t', (df,), q=q)
        var_m, var_s = estimate_var_std(th2)
        cvar_m, cvar_s = estimate_cvar_and_std(th2, xs2, q=q)
        print(f" t(df={df}): VaR_SGLD = {var_m:.4f} (std {var_s:.4f}), "
              f"CVaR_SGLD = {cvar_m:.4f} (std {cvar_s:.4f})")
