import numpy as np
import pandas as pd
from tabulate import tabulate


def softmax(w):
    exp_w = np.exp(w - np.max(w))
    return exp_w / exp_w.sum()


def portfolio_loss(w, x):
    weights = softmax(w)
    return np.dot(weights, x), weights


def grad_portfolio(theta, w, x, q=0.95, gamma=1e-8):
    port_ret, weights = portfolio_loss(w, x)
    indicator = 1.0 if port_ret > theta else 0.0
    g_theta = 1 - (1 / (1 - q)) * indicator + 2 * gamma * theta
    g_w = (1 / (1 - q)) * indicator * weights * (x - port_ret) + 2 * gamma * w
    return g_theta, g_w


def run_sgld(theta0, w0, sampler,
             q=0.95, gamma=1e-8, beta=1e8,
             lr=1e-4, n_iter=1_000_000, burn_in=200_000):

    theta = theta0
    w = np.array(w0, float)
    sqrt_term = np.sqrt(2 * lr / beta)
    thetas, ws = [], []
    for i in range(n_iter):
        x = sampler()
        g_theta, g_w = grad_portfolio(theta, w, x, q, gamma)
        theta += -lr * g_theta + sqrt_term * np.random.randn()
        w += -lr * g_w + sqrt_term * np.random.randn(3)
        if i >= burn_in:
            thetas.append(theta)
            ws.append(w.copy())
    return np.array(thetas), np.vstack(ws)


def sample_sgld(theta0, w0, sampler,
                q=0.95, gamma=1e-8, beta=1e8,
                lr=1e-5, n_iter=200_000):

    return run_sgld(theta0, w0, sampler, q, gamma, beta, lr, n_iter, burn_in=0)


def estimate_var_cvar_from_weights(avg_w, sampler, q=0.95, N=50_000):

    rets = np.array([np.dot(avg_w, sampler()) for _ in range(N)])
    VaR = np.quantile(rets, q)
    CVaR = VaR + np.mean(np.maximum(rets - VaR, 0)) / (1 - q)
    return VaR, CVaR



cases = [
    ("N(500,1)", "N(0,1e6)", "N(0,1e-4)",
     lambda: np.array([
         np.random.normal(500, 1.0),
         np.random.normal(0, 1000.0),
         np.random.normal(0, 0.01)
     ])),
    ("N(500,1)", "N(0,1e6)", "N(0,1)",
     lambda: np.array([
         np.random.normal(500, 1.0),
         np.random.normal(0, 1000.0),
         np.random.normal(0, 1.0)
     ])),
    ("N(0,1e3)", "N(0,1)", "N(0,4)",
     lambda: np.array([
         np.random.normal(0, np.sqrt(1e3)),
         np.random.normal(0, 1.0),
         np.random.normal(0, 2.0)
     ])),
    ("N(0,1)", "N(1,4)", "N(0,1e-4)",
     lambda: np.array([
         np.random.normal(0, 1.0),
         np.random.normal(1, 2.0),
         np.random.normal(0, 0.01)
     ])),
    ("N(0,1)", "N(1,4)", "N(2,1)",
     lambda: np.array([
         np.random.normal(0, 1.0),
         np.random.normal(1, 2.0),
         np.random.normal(2, 1.0)
     ]))
]

rows = []
for X1, X2, X3, sampler in cases:

    thetas_opt, ws_opt = run_sgld(0, [0, 0, 0], sampler)
    theta_star = thetas_opt.mean()
    w_star = ws_opt.mean(axis=0)

    thetas_smpl, ws_smpl = sample_sgld(theta_star, w_star, sampler)
    avg_w = softmax(ws_smpl.mean(axis=0))

    VaR_sgld, CVaR_sgld = estimate_var_cvar_from_weights(avg_w, sampler)

    rows.append({
        "X1": X1, "X2": X2, "X3": X3,
        "w1_SGLD": f"{avg_w[0]:.5f}",
        "w2_SGLD": f"{avg_w[1]:.5f}",
        "w3_SGLD": f"{avg_w[2]:.5f}",
        "VaR_SGLD": f"{VaR_sgld:.3f}",
        "CVaR_SGLD": f"{CVaR_sgld:.3f}"
    })

df = pd.DataFrame(rows)
print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
