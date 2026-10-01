#!/usr/bin/env bash
# Kilpisjarvi AR(5): posterior predictives under the parametric (FDsens) and
# nonparametric (FDsens+) worst-case priors of a shared z-scale
# neighbourhood Gamma_{z,j} = {mu_z in [-mu_z_max, mu_z_max],
# sigma_z in [sigma_z_min, sigma_z_max]}, with the nonparametric radius
# r_j = sup FD over Gamma_{z,j} for every component. Two boxes are run:
#   - mu_z in [-0.4, 0.4], sigma_z in [0.8, 1.25] (paper, r ~= 0.71;
#     outputs keep their original names),
#   - mu_z in [-1, 1],     sigma_z in [0.5, 2]    (r = 25; outputs get the
#     suffix "_mu1_sig0.5-2"), a larger neighbourhood with visibly larger
#     predictive deviations from the reference.
#
# For each box:
#   1. Find both worst-case priors and export their Stan data
#      (also redraws the existing param-vs-nonparam sensitivity plots).
#   2. Sample the AR(5) posterior under each worst-case prior with Stan.
#   3. Plot the reference / parametric / nonparametric posterior predictives.
#
# Fixed seeds throughout (data.seed in the configs, Stan seed below), so
# reruns give the same result. Run from anywhere:
#   bash tests/posteriordb/run_kilpisjarvi_param_vs_nonparam_predictive.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT="tests/posteriordb/run_ark_kilpisjarvi_param_nonparam.py"
STAN_DIR="tests/posteriordb/stan"
OUT_DIR="outputs/paper/results/kilpisjarvi/param_vs_nonparam/stan"
STAN_SEED=27

# "mu_z_max sigma_z_min sigma_z_max" per box.
BOXES=(
  "0.4 0.8 1.25"
  "1 0.5 2"
)

for box in "${BOXES[@]}"; do
  read -r MU_Z_MAX SIGMA_Z_MIN SIGMA_Z_MAX <<< "$box"
  BOX_ARGS=(--mu-z-max "$MU_Z_MAX" --sigma-z-min "$SIGMA_Z_MIN" --sigma-z-max "$SIGMA_Z_MAX")
  TAG="$("$PYTHON" "$SCRIPT" --stage print_tag "${BOX_ARGS[@]}")"

  echo "=== z-box: mu_z in [-$MU_Z_MAX, $MU_Z_MAX], sigma_z in [$SIGMA_Z_MIN, $SIGMA_Z_MAX] (tag '$TAG') ==="

  echo "=== 1/3: worst-case priors (parametric + nonparametric) ==="
  "$PYTHON" "$SCRIPT" --stage optimise "${BOX_ARGS[@]}"

  echo "=== 2/3: Stan sampling under each worst-case prior ==="
  Rscript "$STAN_DIR/sample_ark_stan.R" \
    "$STAN_DIR/kilpisjarvi_ark_param_z.stan" "$OUT_DIR/stan_data_param_z$TAG.json" "$OUT_DIR/draws_param_z$TAG.json" "$STAN_SEED"
  Rscript "$STAN_DIR/sample_ark_stan.R" \
    "$STAN_DIR/kilpisjarvi_ark_kef.stan" "$OUT_DIR/stan_data_kef_z$TAG.json" "$OUT_DIR/draws_kef_z$TAG.json" "$STAN_SEED"

  echo "=== 3/3: posterior predictives ==="
  "$PYTHON" "$SCRIPT" --stage plot_predictive "${BOX_ARGS[@]}"
done
