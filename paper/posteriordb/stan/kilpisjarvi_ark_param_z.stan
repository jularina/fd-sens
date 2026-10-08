// AR(K) model for Kilpisjarvi (centred series) under the parametric
// worst-case prior of the shared z-space neighbourhood, found by
// run_ark_kilpisjarvi.py.
//
// Per component j (alpha, beta[1..K], sigma), with the reference-prior PIT
// z_j = Phi^{-1}(F_ref,j(theta_j)) (exactly N(0, 1) under the reference), the
// candidate is z_j ~ N(mu_z[j], sigma_z[j]^2), i.e.
//   q_j(theta_j) = pi_ref,j(theta_j) * N(z_j; mu_z[j], sigma_z[j]) / N(z_j; 0, 1).
// The PIT Jacobian is shared by q_j and pi_ref,j, so it cancels.
// Data are written by _export_param_z_stan_data.
functions {
  real z_tilt(real z, real mu_z, real sigma_z) {
    return normal_lpdf(z | mu_z, sigma_z) - std_normal_lpdf(z);
  }
}
data {
  int<lower=0> K;
  int<lower=0> T;
  array[T] real y;
  vector[K + 1] mu_ref;             // reference N(mu, sigma) for alpha, beta[1..K]
  vector<lower=0>[K + 1] sigma_ref;
  real<lower=0> gamma_sigma;        // reference HalfCauchy(gamma) for sigma
  real<lower=0, upper=0.5> pit_eps; // clipping of the sigma PIT, as in Python
  vector[K + 2] mu_z;               // rows: alpha, beta[1..K], sigma
  vector<lower=0>[K + 2] sigma_z;
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

  // Gaussian-in-z tilt, component by component.
  target += z_tilt((alpha - mu_ref[1]) / sigma_ref[1], mu_z[1], sigma_z[1]);
  for (k in 1:K)
    target += z_tilt((beta[k] - mu_ref[k + 1]) / sigma_ref[k + 1], mu_z[k + 1], sigma_z[k + 1]);
  {
    real u = fmin(fmax(2 / pi() * atan(sigma / gamma_sigma), pit_eps), 1 - pit_eps);
    target += z_tilt(inv_Phi(u), mu_z[K + 2], sigma_z[K + 2]);
  }

  // Likelihood (unchanged).
  for (t in (K + 1):T) {
    real mu = alpha;
    for (k in 1:K)
      mu += beta[k] * y[t - k];
    y[t] ~ normal(mu, sigma);
  }
}
