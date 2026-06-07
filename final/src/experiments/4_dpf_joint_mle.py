"""Experiment 4: joint MLE of the latent-macro-state choice model by DPF + Adam, comparing how the unidentifiable factor scale is anchored."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.choice_models.macro_deephalo import MacroDeepHalo
from src.datasets.dgp_macro import DGPConfig, data_to_choice_dataset, simulate_macro_choice_dgp
from src.experiments._dpf_common import RESAMPLER_NAMES, build_resampler, psi_prior_from_true_sign
from src.inference import run_dpf_q2_adam
from src.inference.q2.filters import DifferentiableParticleFilter as Q2DPF
from src.inference.q2.filters.common import DTYPE
from src.inference.q2.resampling import SoftResampler
from src.inference.ssm_wrapper import DeepHaloMacroSSM
from src._results import set_all_seeds

DEFAULT_SEEDS = list(range(32, 42))
WILD_INIT = {"mu": 1.0, "phi": 0.5, "sigma": 1.0}
DPF_KW = dict(n_particles=200, n_particles_state=500, n_steps=100, lr=0.01)

# How each design anchors the unidentifiable factor scale (report sec 3.5.3).
DESIGNS = {
    # no anchor: the scale collapses
    "cold":        dict(warm=False, ssm="wild",  free_ssm=True,  free_obs=True,  psi="sign"),
    # recommended: SSM from the proxy (frozen), choice model learned from scratch
    "warm-latent": dict(warm=False, ssm="proxy", free_ssm=False, free_obs=True,  psi="sign"),
    # the reverse: choice model frozen, SSM estimated
    "warm-obs":    dict(warm=True,  ssm="wild",  free_ssm=True,  free_obs=False, psi="sign"),
    # both warm-started, nothing frozen
    "warm-both":   dict(warm=True,  ssm="proxy", free_ssm=True,  free_obs=True,  psi="sign"),
    # sign-flip isolation: SSM at the truth, psi without then with the sign prior
    "SF-noprior":  dict(warm=False, ssm="true",  free_ssm=False, free_obs=True,  psi="none"),
    "SF-prior":    dict(warm=False, ssm="true",  free_ssm=False, free_obs=True,  psi="sign"),
}


def fit_ar1(z):
    """OLS AR(1) fit to a proxy series z. Returns (mu, phi, sigma) as the
    intercept, slope, and residual std of z_t regressed on [1, z_{t-1}]."""
    z0, z1 = z[:-1], z[1:]
    X = np.vstack([np.ones_like(z0), z0]).T
    coef, *_ = np.linalg.lstsq(X, z1, rcond=None)
    return float(coef[0]), float(coef[1]), float(np.std(z1 - X @ coef))


def build_seed(seed, cfg, tau):
    """Build the DGP, the proxy z_t, and the (optionally Stage-2 warm-started)
    model for one seed and design. Returns (data, model, psi_true, x_true, ssm_init)."""
    set_all_seeds(seed)
    data = simulate_macro_choice_dgp(
        scenario="B", cfg=DGPConfig(T=200, N_t=50, K_min=2, K_max=5), seed=seed)
    psi_true, x_true, M, tp = data["_psi_offer"], data["_x_true"], data["M"], data["_true_params"]
    z = (x_true + np.random.default_rng(seed + 100).normal(0, tau, x_true.shape)).astype(np.float32)
    psi_init = np.zeros(M, np.float32) if cfg["psi"] == "none" else psi_prior_from_true_sign(psi_true)
    model = MacroDeepHalo(M=M, psi_init=psi_init, epochs=50 if cfg["warm"] else 0)
    if cfg["warm"]:
        model.fit(data_to_choice_dataset(data, x_t_override=z), verbose=0)
    if cfg["ssm"] == "true":
        ssm_init = {"mu": tp["mu"], "phi": tp["phi"], "sigma": tp["sigma"]}
    elif cfg["ssm"] == "proxy":
        mz, pz, sz = fit_ar1(z)
        ssm_init = {"mu": mz, "phi": pz, "sigma": sz}
    else:
        ssm_init = dict(WILD_INIT)
    return data, model, psi_true, x_true, ssm_init


def run_one(seed, cfg, tau=0.2, resampler=None):
    """Train one design on one seed and return a dict of recovery metrics.

    The choice metrics score the learned choice model with the filtered state
    x_hat: rmse_p against the DGP's true probabilities and nll against the
    observed choices. They reflect the joint choice-and-state quality the design
    would deploy.
    """
    data, model, psi_true, x_true, ssm_init = build_seed(seed, cfg, tau)
    res = run_dpf_q2_adam(
        model, data, init=ssm_init,
        free_mu=cfg["free_ssm"], free_phi=cfg["free_ssm"], free_sigma=cfg["free_ssm"],
        free_dh=cfg["free_obs"], free_psi=cfg["free_obs"],
        seed=seed, verbose=False, resampler=resampler or SoftResampler(alpha=0.5), **DPF_KW)
    psi_hat, x_hat = model.psi_offer.numpy(), res["x_hat"]
    ds = data_to_choice_dataset(data, x_t_override=x_hat)
    pred = model.predict_probas(ds).numpy()
    true_p = data["_true_probs"].reshape(-1, data["M"])
    slate_size = (true_p > 0).sum(axis=1)
    return {
        "mu": res["mu"], "phi": res["phi"], "sigma": res["sigma"],
        "psi_pearson": float(np.corrcoef(psi_hat, psi_true)[0, 1]),
        "psi_rmse": float(np.sqrt(np.mean((psi_hat - psi_true) ** 2))),
        "nll": float(model.evaluate(ds)),
        "rmse_p": float(np.sqrt(np.mean(((pred - true_p) ** 2).sum(axis=1) / slate_size))),
        "rmse_x": float(np.sqrt(np.mean((x_hat - x_true) ** 2))),
        "ess": float(np.mean(res["ess"])),
    }


def _mean_std(rows, k):
    v = np.array([r[k] for r in rows])
    return f"{v.mean():+.3f}+/-{v.std():.3f}"


def _summarize(name, rows):
    npos = sum(1 for r in rows if r["psi_pearson"] > 0)
    pa = np.abs([r["psi_pearson"] for r in rows]).mean()
    print(f"RESULT >> {name:11s} MEAN mu={_mean_std(rows,'mu')} phi={_mean_std(rows,'phi')} "
          f"sig={_mean_std(rows,'sigma')} psiP={_mean_std(rows,'psi_pearson')} (|{pa:.3f}|, {npos}+/{len(rows)-npos}-) "
          f"psiRMSE={_mean_std(rows,'psi_rmse')} NLL={_mean_std(rows,'nll')} RMSEp={_mean_std(rows,'rmse_p')} "
          f"RMSEx={_mean_std(rows,'rmse_x')}", flush=True)


def run_designs(seeds, names):
    for name in names:
        rows = []
        for s in seeds:
            r = run_one(s, DESIGNS[name])
            rows.append(r)
            print(f"RESULT {name:11s} seed={s} mu={r['mu']:+.3f} phi={r['phi']:.3f} "
                  f"sig={r['sigma']:.3f} psiP={r['psi_pearson']:+.3f} psiRMSE={r['psi_rmse']:.3f} "
                  f"NLL={r['nll']:.3f} RMSEp={r['rmse_p']:.3f} RMSEx={r['rmse_x']:.3f}", flush=True)
        _summarize(name, rows)


def run_proxy_sweep(seeds, taus):
    for tau in taus:
        _summarize(f"warm-latent t{tau}", [run_one(s, DESIGNS["warm-latent"], tau=tau) for s in seeds])


def run_resampler_sweep(seeds):
    for design in ("warm-latent", "cold"):
        for rname in RESAMPLER_NAMES:
            rows = [run_one(s, DESIGNS[design], resampler=build_resampler(rname)) for s in seeds]
            sig = np.mean([r["sigma"] for r in rows])
            rmse = np.mean([r["rmse_x"] for r in rows])
            ess = np.mean([r["ess"] for r in rows])
            print(f"RESAMP {design:8s} {rname:16s} sig={sig:.2f} rmse={rmse:.2f} ess={ess:.0f}", flush=True)


def measure_gradient_variance(design, n_rng=20):
    """Train the design to convergence (seed 32), then measure the cross-rng SD of
    the FIVO gradient at that operating point, per resampler, for the parameters
    the design trains."""
    cfg = DESIGNS[design]
    data, model, psi_true, x_true, ssm_init = build_seed(32, cfg, 0.2)
    res = run_dpf_q2_adam(
        model, data, init=ssm_init,
        free_mu=cfg["free_ssm"], free_phi=cfg["free_ssm"], free_sigma=cfg["free_ssm"],
        free_dh=cfg["free_obs"], free_psi=cfg["free_obs"],
        seed=32, verbose=False, resampler=SoftResampler(alpha=0.5), **DPF_KW)
    meas = ({"mu": res["mu"], "phi": res["phi"], "sigma": res["sigma"]}
            if cfg["free_ssm"] else ssm_init)
    for rname in RESAMPLER_NAMES:
        mu_v = tf.Variable(meas["mu"], dtype=DTYPE)
        phi_v = tf.Variable(np.arctanh(meas["phi"]), dtype=DTYPE)
        ls_v = tf.Variable(np.log(meas["sigma"]), dtype=DTYPE)
        scal_vars, scal_tags = [], []
        if cfg["free_ssm"]:
            scal_vars, scal_tags = [mu_v, phi_v, ls_v], ["mu", "phi_raw", "log_sigma"]
        if cfg["free_obs"]:
            psi_v = tf.Variable(tf.cast(model.psi_offer, DTYPE).numpy(), dtype=DTYPE)
            dh_w = list(model.deephalo.trainable_weights)
            free = scal_vars + [psi_v] + dh_w
        else:
            psi_v = tf.constant(tf.cast(model.psi_offer, DTYPE).numpy(), dtype=DTYPE)
            dh_w, free = [], scal_vars
        ssm = DeepHaloMacroSSM(model, data, mu_var=mu_v, phi_raw_var=phi_v,
                               log_sigma_var=ls_v, psi_var=psi_v, precompute_h=not cfg["free_obs"])
        dpf = Q2DPF(n_particles=200, resampler=build_resampler(rname))
        ys = ssm.ys_indices
        grads = []
        for k in range(n_rng):
            rng = tf.random.Generator.from_seed(1000 + k)
            with tf.GradientTape() as tape:
                loss = -dpf.filter(ssm, ys, rng=rng).diagnostics["log_marginal_likelihood"]
            grads.append(tape.gradient(loss, free))
        out, i = {}, 0
        for t in scal_tags:
            out[t] = float(np.std([float(grads[k][i].numpy()) for k in range(n_rng)]))
            i += 1
        if cfg["free_obs"]:
            arr = np.stack([grads[k][i].numpy() for k in range(n_rng)], 0)
            out["psi"] = float(np.mean(np.std(arr, 0)))
            i += 1
            sds = [np.std(np.stack([grads[k][i + w].numpy().ravel() for k in range(n_rng)], 0), 0)
                   for w in range(len(dh_w))]
            out["dh"] = float(np.mean(np.concatenate(sds)))
        print(f"GRADSD {design:8s} {rname:16s} " + "  ".join(f"{k}={v:.4g}" for k, v in out.items()), flush=True)


def main():
    """Parse CLI args and run the selected Experiment 4 part."""
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="designs",
                   choices=["designs", "signflip", "proxy", "gradvar", "resampler", "all"])
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--taus", default="0.2,0.5,1.0")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    taus = [float(t) for t in args.taus.split(",")]

    if args.part in ("designs", "all"):
        run_designs(seeds, ["cold", "warm-latent", "warm-obs", "warm-both"])
    if args.part in ("signflip", "all"):
        run_designs(seeds, ["SF-noprior", "SF-prior"])
    if args.part in ("proxy", "all"):
        run_proxy_sweep(seeds, taus)
    if args.part in ("gradvar", "all"):
        for design in ("warm-latent", "warm-obs"):
            measure_gradient_variance(design)
    if args.part in ("resampler", "all"):
        run_resampler_sweep(seeds)


if __name__ == "__main__":
    main()
