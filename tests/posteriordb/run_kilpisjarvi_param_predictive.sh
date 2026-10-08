#!/usr/bin/env bash
# Kilpisjarvi AR(5), parametric only: z-scale sensitivity figures and
# posterior predictives under the parametric worst-case prior of the shared
# z-scale box Gamma_z = {mu_z in [-MU_Z_MAX, MU_Z_MAX],
# sigma_z in [SIGMA_Z_MIN, SIGMA_Z_MAX]}.
#
#   1. Worst-case corners, kilpisjarvi_param_component_sensitivity.pdf,
#      kilpisjarvi_param_three_panel_priors.pdf, and the Stan data.
#   2. Stan sampling of the AR(5) posterior under the worst-case prior.
#   3. kilpisjarvi-posterior-predictive-{ref,corner}.pdf and
#      kilpisjarvi-acf-comparison.pdf.
#
# Box via positional args (defaults = paper box, as in configs/paper/real/ark_kilpisjarvi.yaml):
#   bash tests/posteriordb/run_kilpisjarvi_param_predictive.sh [MU_Z_MAX SIGMA_Z_MIN SIGMA_Z_MAX]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT="tests/posteriordb/run_ark_kilpisjarvi.py"
STAN_DIR="tests/posteriordb/stan"
OUT_DIR="outputs/paper/results/kilpisjarvi/param/stan"
STAN_SEED=27

MU_Z_MAX="${1:-1}"
SIGMA_Z_MIN="${2:-0.5}"
SIGMA_Z_MAX="${3:-2}"
BOX_ARGS=(playground.z_mu_max="$MU_Z_MAX" playground.z_sigma_min="$SIGMA_Z_MIN" playground.z_sigma_max="$SIGMA_Z_MAX")
# Same as _z_box_tag in run_ark_kilpisjarvi.py.
TAG="$("$PYTHON" -c "import sys; m, a, b = map(float, sys.argv[1:]); print(f'_mu{m:g}_sig{a:g}-{b:g}')" \
  "$MU_Z_MAX" "$SIGMA_Z_MIN" "$SIGMA_Z_MAX")"

echo "=== z-box: mu_z in [-$MU_Z_MAX, $MU_Z_MAX], sigma_z in [$SIGMA_Z_MIN, $SIGMA_Z_MAX] (tag '$TAG') ==="

echo "=== 1/3: worst-case corners, sensitivity/prior figures, Stan data ==="
"$PYTHON" "$SCRIPT" playground.stage=optimise "${BOX_ARGS[@]}"

echo "=== 2/3: Stan sampling under the worst-case prior ==="
Rscript "$STAN_DIR/sample_ark_stan.R" \
  "$STAN_DIR/kilpisjarvi_ark_param_z.stan" "$OUT_DIR/stan_data_param_z$TAG.json" "$OUT_DIR/draws_param_z$TAG.json" "$STAN_SEED"

echo "=== 3/3: posterior predictives ==="
"$PYTHON" "$SCRIPT" playground.stage=plot_predictive "${BOX_ARGS[@]}"
