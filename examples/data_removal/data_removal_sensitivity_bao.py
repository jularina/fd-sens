"""FD sensitivity of the DESI DR2 BAO LCDM posterior to removing whole BAO blocks.

The 13 measurements form 7 independent Gaussian blocks, so the per-block scores
gamma_g(theta) = J_g(theta)^T Sigma_g^{-1} (D_g - M_g(theta)) are passed as `data_grad` and every subset of at most
`max_removed` blocks is evaluated from a single run of the sampler.
"""
import os
import time

from interpretation.plots import (plot_removal_budget, plot_removal_interactions, plot_removal_single,
                                  save_sensitivity_result)
from paper.bao.bao import BLOCK_NAMES, G, MODELS, block_scores, run_posterior
from src.common.bayesian_model.samples import PosteriorSamplesModel
from src.common.utils.files_operations import load_plot_config
from src.data_removal.sensitivity import data_removal_sensitivity

OUTPUT_DIR = "outputs/examples/data_removal/bao_LCDM"
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
# Fonts, LaTeX rendering, figure size and the data-removal colours of the paper figures.
PLOT_CFG = load_plot_config(os.path.join(REPO_ROOT, "configs", "plots", "overleaf_plots_settings.yaml"))

# 1. Reference posterior: all blocks kept.
t0 = time.time()
posterior, acc = run_posterior("LCDM", nwalkers=64, nsteps=4000, burn=1500, thin=10, seed=1)
print(f"Reference posterior: {posterior.shape[0]} draws of {MODELS['LCDM']['names']}, "
      f"acceptance {acc:.2f}, {time.time() - t0:.0f}s")

# 2. Wrap the draws with the block scores, shape (m, G, d).
model = PosteriorSamplesModel(posterior_samples=posterior, data_grad=block_scores)

# 3. Remove at most G - 1 = 6 of the 7 blocks (alpha=0.5 would allow floor(0.5 * 7) = 3).
#    whiten=True measures the FD in posterior-standardised coordinates, so omega_b, omega_c and H0 contribute on
#    comparable scales.
result = data_removal_sensitivity(model, max_removed=G - 1, names=BLOCK_NAMES, whiten=True, keep_all=True)
print(result)

# 4. Interpret: single-block FDs, block interactions, and the optimal removal per budget.
save_sensitivity_result(result, OUTPUT_DIR)
plot_removal_single(result, OUTPUT_DIR, file_stem="bao_removal_single", plot_cfg=PLOT_CFG)
plot_removal_interactions(result, OUTPUT_DIR, file_stem="bao_removal_interactions", plot_cfg=PLOT_CFG)
plot_removal_budget(result, OUTPUT_DIR, file_stem="bao_removal_budget", plot_cfg=PLOT_CFG)
print(f"Saved the result and plots to {OUTPUT_DIR}/")
