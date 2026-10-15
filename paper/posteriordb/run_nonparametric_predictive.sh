#!/usr/bin/env bash
# Kilpisjarvi AR(5), nonparametric only: posterior predictive under the
# worst-case KEF prior at the shared radius r_j = sup FD over the z-scale box
# Gamma_z = {mu_z in [-MU_Z_MAX, MU_Z_MAX], sigma_z in [SIGMA_Z_MIN, SIGMA_Z_MAX]}.
#
#   1. Worst-case KEF prior per component and its Stan data.
#   2. Stan sampling of the AR(5) posterior under the worst-case prior.
#   3. kilpisjarvi_nonparam_posterior_predictive_ref.pdf and
#      kilpisjarvi_posterior_predictive_worst_case.pdf.
#
# Box via positional args (defaults = paper box, r_j ~= 0.71, as in configs/paper/real/ark_kilpisjarvi_nonparam.yaml):
#   bash paper/posteriordb/run_nonparametric_predictive.sh [MU_Z_MAX SIGMA_Z_MIN SIGMA_Z_MAX]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT="paper/posteriordb/run_nonparametric.py"
STAN_DIR="paper/posteriordb/stan"
OUT_DIR="outputs/paper/results/kilpisjarvi/nonparam/stan"
STAN_SEED=27

MU_Z_MAX="${1:-0.4}"
SIGMA_Z_MIN="${2:-0.8}"
SIGMA_Z_MAX="${3:-1.25}"
BOX_ARGS=(playground.z_mu_max="$MU_Z_MAX" playground.z_sigma_min="$SIGMA_Z_MIN" playground.z_sigma_max="$SIGMA_Z_MAX")
# Same as _z_box_tag in run_parametric.py.
TAG="$("$PYTHON" -c "import sys; m, a, b = map(float, sys.argv[1:]); print(f'_mu{m:g}_sig{a:g}-{b:g}')" \
  "$MU_Z_MAX" "$SIGMA_Z_MIN" "$SIGMA_Z_MAX")"

echo "=== z-box: mu_z in [-$MU_Z_MAX, $MU_Z_MAX], sigma_z in [$SIGMA_Z_MIN, $SIGMA_Z_MAX] (tag '$TAG') ==="

echo "=== 1/3: worst-case KEF prior, Stan data ==="
"$PYTHON" "$SCRIPT" playground.stage=optimise "${BOX_ARGS[@]}"

echo "=== 2/3: Stan sampling under the worst-case prior ==="
Rscript "$STAN_DIR/sample_ark_stan.R" \
  "$STAN_DIR/kilpisjarvi_ark_kef.stan" "$OUT_DIR/stan_data_kef_z$TAG.json" "$OUT_DIR/draws_kef_z$TAG.json" "$STAN_SEED"

echo "=== 3/3: posterior predictives ==="
"$PYTHON" "$SCRIPT" playground.stage=plot_predictive "${BOX_ARGS[@]}"
