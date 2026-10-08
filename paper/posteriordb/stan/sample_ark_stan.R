library(rstan)
library(jsonlite)

# Samples an AR(K) Stan model and saves its post-warmup draws as a JSON list
# of chains ({"alpha": [...], "beta[1]": [...], ..., "sigma": [...]} per
# chain), the format paper/posteriordb/run_parametric.py's
# _load_corner_draws_json / _stack_corner_chains read.
#
# Usage: Rscript sample_ark_stan.R <stan_file> <data_json> <out_json> [seed]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript sample_ark_stan.R <stan_file> <data_json> <out_json> [seed]")
stan_file <- args[1]
data_path <- args[2]
out_path <- args[3]
seed <- if (length(args) >= 4) as.integer(args[4]) else 27L


save_posterior_chains_json <- function(fit, json_path, K) {
  sims <- rstan::extract(fit, permuted = FALSE, inc_warmup = FALSE)
  # sims dims: iterations x chains x parameters_in_flat_order
  parnames <- dimnames(sims)$parameters
  n_chains <- dim(sims)[2]

  needed <- c("alpha", paste0("beta[", 1:K, "]"), "sigma")
  missing <- setdiff(needed, parnames)
  if (length(missing) > 0) stop(sprintf("Parameters not found: %s", paste(missing, collapse = ", ")))

  chains_list <- vector("list", n_chains)
  for (ch in seq_len(n_chains)) {
    chain_dict <- list()
    for (p in needed) {
      chain_dict[[p]] <- as.numeric(sims[, ch, match(p, parnames)])
    }
    chains_list[[ch]] <- chain_dict
  }
  dir.create(dirname(json_path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(chains_list, path = json_path, pretty = FALSE, auto_unbox = TRUE, digits = NA)
}


stan_data <- jsonlite::fromJSON(data_path)  # matrices come back as (K+2) x M

start <- Sys.time()
fit <- stan(
  file = stan_file,
  data = stan_data,
  chains = 5,        # as for the reference posterior: 5 chains,
  warmup = 1000,     # 1000 burn-in iterations each,
  iter = 2000,       # 5 x 1000 = 5000 post-warmup draws in total
  cores = 1,
  seed = seed,
  refresh = 0
)
elapsed_secs <- as.numeric(difftime(Sys.time(), start, units = "secs"))
cat(sprintf("%s: sampling took %.1f s\n", basename(stan_file), elapsed_secs))
print(fit, pars = c("alpha", "beta", "sigma"))

save_posterior_chains_json(fit, out_path, K = stan_data$K)
cat(sprintf("Saved draws to %s\n", out_path))
