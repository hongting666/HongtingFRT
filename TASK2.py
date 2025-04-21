import numpy as np
import pandas as pd
from scipy.stats import norm
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
    g_theta = 1 - (1/(1-q))*indicator + 2*gamma*theta
    g_w = (1/(1-q))*indicator * weights * (x - port_ret) + 2*gamma*w
    return g_theta, g_w

def run_sgld(theta0, w0, sampler, q=0.95, gamma=1e-8, beta=1e8, lr=1e-4, n_iter=5000, burn_in=500):
    theta = theta0
    w = np.array(w0, float)
    sqrt_term = np.sqrt(2 * lr / beta)
    thetas, ws = [], []
    for i in range(n_iter):
        x = sampler()
        g_theta, g_w = grad_portfolio(theta, w, x, q, gamma)
        theta += -lr*g_theta + sqrt_term*np.random.randn()
        w += -lr*g_w + sqrt_term*np.random.randn(3)
        if i >= burn_in:
            thetas.append(theta)
            ws.append(w.copy())
    return np.array(thetas), np.vstack(ws)

def estimate_cvar(theta_samples, w_samples, sampler, q=0.95, gamma=1e-8, N=200):
    cvars = []
    for th, w in zip(theta_samples, w_samples):
        losses = []
        for _ in range(N):
            x = sampler()
            ret, _ = portfolio_loss(w, x)
            losses.append(max(ret - th, 0))
        cvars.append(th + np.mean(losses)/(1-q) + gamma*(th**2 + np.sum(w**2)))
    return np.array(cvars)

cases = [
    ("N(500,1)", "N(0,1e6)", "N(0,1e-4)", lambda: np.array([np.random.normal(500,1), np.random.normal(0,1e6), np.random.normal(0,1e-4)])),
    ("N(500,1)", "N(0,1e6)", "N(0,1)",   lambda: np.array([np.random.normal(500,1), np.random.normal(0,1e6), np.random.normal(0,1)])),
    ("N(0,1e3)",  "N(0,1)",    "N(0,4)",   lambda: np.array([np.random.normal(0,1e3), np.random.normal(0,1), np.random.normal(0,4)])),
    ("N(0,1)",    "N(1,4)",    "N(0,1e-4)",lambda: np.array([np.random.normal(0,1), np.random.normal(1,4), np.random.normal(0,1e-4)])),
    ("N(0,1)",    "N(1,4)",    "N(2,1)",   lambda: np.array([np.random.normal(0,1), np.random.normal(1,4), np.random.normal(2,1)]))
]

rows = []
for X1, X2, X3, sampler in cases:
    thetas, ws = run_sgld(0, [0,0,0], sampler, q=0.95, n_iter=5000, burn_in=500)
    cvars = estimate_cvar(thetas, ws, sampler, q=0.95, N=200)
    row = {
        "X1": X1, "X2": X2, "X3": X3,
        "w1": f"{softmax(np.mean(ws,0))[0]:.4f}",
        "w2": f"{softmax(np.mean(ws,0))[1]:.4f}",
        "w3": f"{softmax(np.mean(ws,0))[2]:.4f}",
        "VaR_SGLD": f"{np.mean(thetas):.3f}\n({np.std(thetas):.4f})",
        "CVaR_SGLD": f"{np.mean(cvars):.3f}\n({np.std(cvars):.4f})"
    }
    rows.append(row)

df = pd.DataFrame(rows)

print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

