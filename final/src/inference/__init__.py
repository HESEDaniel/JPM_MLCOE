"""Production inference --- thin wrappers around the Q2 paper PF / DPF.

Public API:
  - ``run_pf_q2``       - bootstrap PF, returns moments dict
  - ``run_dpf_q2_adam`` - DPF + Adam joint MLE of (mu, phi, sigma, psi, theta_DH)
"""

from .pf import run_pf_q2
from .dpf import run_dpf_q2_adam

__all__ = ["run_pf_q2", "run_dpf_q2_adam"]
