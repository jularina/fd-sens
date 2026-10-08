// AR(K) model for Kilpisjarvi (centred series) under the worst-case
// nonparametric KEF prior found by run_ark_kilpisjarvi_nonparam.py.
//
// Per component j (alpha, beta[1..K], sigma) the KEF prior is
//   q_j(theta_j) propto pi_ref,j(theta_j) * exp(f_j(z_j(theta_j))),
//   f_j(z) = sum_m lambda[j, m] * phi(|z - centres[j, m]|),
// where z_j = Phi^{-1}(F_ref,j(theta_j)) is the reference-prior PIT
// (exactly N(0, 1) under the reference) and phi is the Matern-7/2 kernel.
// The PIT Jacobian is shared by q_j and pi_ref,j, so it cancels, and the
// normalising constant Z_j is a constant, so neither is needed here.
// All data (centres, lambda rescaled to r_pred, lengthscales) are written by
// _export_kef_stan_data in run_ark_kilpisjarvi_nonparam.py.
functions {
  real kef_tilt(real z, row_vector c, row_vector lam, real ell, real var_k) {
    real f = 0;
    for (m in 1:cols(c)) {
      real x = sqrt(7.0) * abs(z - c[m]) / ell;
      f += lam[m] * var_k * (1 + x + 0.4 * x^2 + x^3 / 15) * exp(-x);
    }
    return f;
  }
}
data {
  int<lower=0> K;
  int<lower=0> T;
  array[T] real y;
  int<lower=1> M;                   // number of basis centres per component
  vector[K + 1] mu_ref;             // reference N(mu, sigma) for alpha, beta[1..K]
  vector<lower=0>[K + 1] sigma_ref;
  real<lower=0> gamma_sigma;        // reference HalfCauchy(gamma) for sigma
  real<lower=0, upper=0.5> pit_eps; // clipping of the sigma PIT, as in Python
  matrix[K + 2, M] centres;         // rows: alpha, beta[1..K], sigma (z-space)
  matrix[K + 2, M] lambda;          // lambda_star rescaled to r_pred
  vector<lower=0>[K + 2] ell;
  vector<lower=0>[K + 2] var_k;
  real r_pred;                      // for bookkeeping only
}
parameters {
  real alpha;
  array[K] real beta;
  real<lower=0> sigma;
}
model {
  // Reference prior.
  alpha ~ normal(mu_ref[1], sigma_ref[1]);
  for (k in 1:K)
    beta[k] ~ normal(mu_ref[k + 1], sigma_ref[k + 1]);
  sigma ~ cauchy(0, gamma_sigma);

  // KEF tilt, component by component in z-space.
  target += kef_tilt((alpha - mu_ref[1]) / sigma_ref[1],
                     centres[1], lambda[1], ell[1], var_k[1]);
  for (k in 1:K)
    target += kef_tilt((beta[k] - mu_ref[k + 1]) / sigma_ref[k + 1],
                       centres[k + 1], lambda[k + 1], ell[k + 1], var_k[k + 1]);
  {
    real u = fmin(fmax(2 / pi() * atan(sigma / gamma_sigma), pit_eps), 1 - pit_eps);
    target += kef_tilt(inv_Phi(u), centres[K + 2], lambda[K + 2], ell[K + 2], var_k[K + 2]);
  }

  // Likelihood (unchanged).
  for (t in (K + 1):T) {
    real mu = alpha;
    for (k in 1:K)
      mu += beta[k] * y[t - k];
    y[t] ~ normal(mu, sigma);
  }
}
