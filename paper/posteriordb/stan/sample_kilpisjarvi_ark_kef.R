library(rstan)
library(jsonlite)

# Samples the Kilpisjarvi AR(K) posterior under the worst-case nonparametric
# KEF prior, for each radius r_pred whose Stan data was exported by
# paper/posteriordb/run_ark_kilpisjarvi_nonparam.py (_export_kef_stan_data).
# Run from the fd-sens repo root.

stan_file <- "paper/posteriordb/stan/kilpisjarvi_ark_kef.stan"
stan_data_dir <- "outputs/paper/results/kilpisjarvi/nonparam/stan"
draws_dir <- "/Users/arinaodv/Desktop/folder/study_phd/code/posteriordb/posterior_database/reference_posteriors/draws/draws"
radii <- c(5, 20, 50, 100, 200, 500)

model <- stan_model(file = stan_file)


save_posterior_chains_json <- function(fit, json_path, K = 5) {
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
  jsonlite::write_json(chains_list, path = json_path, pretty = FALSE, auto_unbox = TRUE, digits = NA)
}


for (r in radii) {
  data_path <- file.path(stan_data_dir, sprintf("kilpisjarvi_kef_stan_data_r%g.json", r))
  kef_data <- jsonlite::fromJSON(data_path)  # centres / lambda come back as (K+2) x M matrices

  start <- Sys.time()
  fit <- sampling(
    model,
    data = kef_data,
    chains = 10,       # number of Markov chains
    warmup = 1000,     # number of warmup iterations per chain
    iter = 10000,      # total number of iterations per chain
    cores = 1,         # number of cores (could use one per chain)
    seed = 27,
    refresh = 0        # no progress shown
  )
  elapsed_secs <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  cat(sprintf("r = %g: %.1f s\n", r, elapsed_secs))
  print(fit, pars = c("alpha", "beta", "sigma"))

  save_posterior_chains_json(
    fit,
    file.path(draws_dir, sprintf("kilpisjarvi-reference-draws-kef-r%g.json", r)),
    K = kef_data$K
  )
}
