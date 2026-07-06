from typing import Dict, List, Tuple, Any
import itertools
import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull
from tqdm import tqdm
from scipy.optimize import differential_evolution, dual_annealing
from dataclasses import dataclass

from distributions.inverse_wishart import InverseWishart
from src.distributions.gaussian import Gaussian, MultivariateGaussian
from src.distributions.gamma import Gamma
from src.optimization.corners import get_corners


@dataclass
class BlackBoxOptResult:
    eta_sup: np.ndarray
    val_sup: float
    eta_inf: np.ndarray
    val_inf: float
    S_hat: float
    nfev_sup: int
    nfev_inf: int


@dataclass
class BlackBoxCopulaOptResult:
    lambda_sup: float
    val_sup: float
    lambda_inf: float
    val_inf: float
    S_hat: float
    nfev_sup: int
    nfev_inf: int


class OptimizationCornerPointsBase:
    def __init__(
        self,
        posterior_estimator,
        prior_config: Dict,
        loss_config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Base class to handle parametric quadratic form optimization.
        """
        self.posterior_estimator = posterior_estimator
        self.model = posterior_estimator.model
        self.distribution_cls = distribution_cls

        self.param_ranges: Dict[str, Tuple[float, float]] = prior_config["parameters_box_range"]["ranges"]
        self.param_nums: Dict[str, int] = prior_config["parameters_box_range"]["nums"]
        self.param_names = list(self.param_ranges.keys())

        self.lr_ranges: Dict[str, Tuple[float, float]] = loss_config["parameters_box_range"]["ranges"]["lr"]
        self.lr_nums: Dict[str, int] = loss_config["parameters_box_range"]["nums"]["lr"]

        self.Lambda_prior, self.b_prior, self.c_prior = self.posterior_estimator.compute_fisher_quadratic_form_prior_only()

        self.parameter_grid = self._generate_full_parameter_grid()
        self.distribution_corner_points: List[Dict[str, float]] = self._generate_corner_points()
        self.lr_grid = self._generate_full_lr_grid()
        self.lr_corners = self._generate_lr_corners()

        self.Lambda_loss, self.b_loss, self.c_loss = self.posterior_estimator.compute_fisher_quadratic_form_lr_only()

    def _generate_corner_points(self):
        eta_list = []
        dists = []
        for v in self.parameter_grid.values():
            eta_list.append(np.asarray(v["natural_parameters"], dtype=float))
            dists.append(v["distribution"])

        all_eta = np.vstack(eta_list)
        hull = ConvexHull(all_eta)
        vertex_idxs = sorted(set(hull.vertices.tolist()))
        return [dists[i] for i in vertex_idxs]

    def _evaluate_prior_qf(self, eta_tilde: np.ndarray) -> float:
        return eta_tilde @ self.Lambda_prior @ eta_tilde + self.b_prior @ eta_tilde + self.c_prior

    def _generate_full_parameter_grid(self) -> Dict:
        pass

    def _generate_full_lr_grid(self):
        lr_grid = np.linspace(self.lr_ranges[0], self.lr_ranges[-1], self.lr_nums).tolist()

        return lr_grid

    def _generate_lr_corners(self):
        return [self.lr_ranges[0], self.lr_ranges[-1]]

    def evaluate_all_prior_corners(self) -> Tuple:
        results = []

        for corner_distribution in self.distribution_corner_points:
            params = corner_distribution.parameters_dict
            self.model.set_prior_parameters(params, distribution_cls=self.distribution_cls)
            eta = self.model.prior.natural_parameters()
            est = self._evaluate_prior_qf(eta)
            results.append((params, eta, est))

        results.sort(key=lambda x: x[2], reverse=True)
        self.model.back_to_prior_candidate()

        return results, results[0][0]

    def evaluate_all_prior_combinations(self) -> List:
        results = []

        for values in self.parameter_grid:
            param_dict = dict(zip(self.param_names, values))
            self.model.set_prior_parameters(param_dict, distribution_cls=self.distribution_cls)
            eta = self.model.prior.natural_parameters()
            est = self._evaluate_prior_qf(eta)
            results.append((param_dict, eta, est))
            print(f"Corner: {param_dict} => Estimated obj: {est:.6f}")

        results.sort(key=lambda x: x[2], reverse=True)
        self.model.back_to_prior_candidate()

        return results

    def evaluate_all_lr_corners(self) -> List:
        results = []

        for lr_corner in self.lr_corners:
            self.model.set_lr_parameter(lr_corner)
            lr = self.model.loss_lr
            est = lr**2 * self.Lambda_loss + self.b_loss * lr + self.c_loss
            results.append((lr_corner, est))
            print(f"Corner: {lr_corner} => Estimated obj: {est:.6f}")

        results.sort(key=lambda x: x[1], reverse=True)
        self.model.back_to_lr_init()

        return results

    def evaluate_full_lr_grid(self) -> List:
        results = []

        for lr in self.lr_grid:
            self.model.set_lr_parameter(lr)
            lr = self.model.loss_lr
            est = lr**2 * self.Lambda_loss + self.b_loss * lr + self.c_loss
            results.append((lr, est))
            print(f"Corner: {lr} => Estimated obj: {est:.6f}")

        results.sort(key=lambda x: x[1], reverse=True)
        self.model.back_to_lr_init()

        return results


class OptimizationCornerPointsUnivariateGaussian(OptimizationCornerPointsBase):
    def __init__(
        self,
        posterior_estimator,
        prior_config: Dict,
        loss_config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Grid/corner quadratic form generation and optimization for univariate gaussian.
        """
        super().__init__(posterior_estimator=posterior_estimator, prior_config=prior_config,
                         loss_config=loss_config, distribution_cls=distribution_cls)

    def _generate_mu_grid(self) -> List:
        mu_ranges = self.param_ranges['mu']
        mu_num = self.param_nums['mu']
        mu_grid = np.linspace(mu_ranges[0], mu_ranges[-1], mu_num).tolist()

        return mu_grid

    def _generate_sigma_grid(self) -> List[np.ndarray]:
        sigma_ranges = self.param_ranges['sigma']
        sigma_num = self.param_nums['sigma']
        sigma_grid = np.linspace(sigma_ranges[0], sigma_ranges[-1], sigma_num).tolist()

        return sigma_grid

    def _generate_full_parameter_grid(self) -> Dict:
        mu_grid = self._generate_mu_grid()
        sigma_grid = self._generate_sigma_grid()
        parameter_grid = {}

        for mu, sigma in itertools.product(mu_grid, sigma_grid):
            try:
                dist = self.distribution_cls(mu=mu, sigma=sigma)
            except Exception as e:
                print(f"Exception: {e} while initializing the distribution with mu={mu}, aigma={sigma}.")
                continue

            augmented_eta = dist.augmented_natural_parameters()
            eta = dist.natural_parameters()
            parameter_grid[(mu, sigma)] = {
                "augmented_natural_parameters": augmented_eta,
                "natural_parameters": eta,
                "distribution": dist
            }

        return parameter_grid


class OptimizationCornerPointsUnivariateGaussianConjugate(OptimizationCornerPointsUnivariateGaussian):
    def __init__(
        self,
        posterior_estimator,
        prior_config: Dict,
        loss_config: Dict,
        distribution_cls=Gaussian,
    ):
        """
        Same grid/corner structure as OptimizationCornerPointsUnivariateGaussian but
        evaluates the prior objective using the exact conjugate-Gaussian closed-form FD
        instead of the quadratic-form approximation.
        """
        super().__init__(
            posterior_estimator=posterior_estimator,
            prior_config=prior_config,
            loss_config=loss_config,
            distribution_cls=distribution_cls,
        )

    def _evaluate_prior_qf(self, eta_tilde: np.ndarray) -> float:
        # model.prior has already been set via set_prior_parameters; align prior_candidate
        self.model.prior_candidate = self.model.prior
        return self.posterior_estimator.estimate_fisher_for_gaussians()


class OptimizationCornerPointsMultivariateGaussian(OptimizationCornerPointsBase):
    def __init__(
        self,
        posterior_estimator,
        prior_config: Dict,
        loss_config: Dict,
        distribution_cls=MultivariateGaussian,
    ):
        """
        Grid/corner quadratic form generation and optimization for multivariate gaussian.
        """
        super().__init__(posterior_estimator=posterior_estimator, prior_config=prior_config,
                         loss_config=loss_config, distribution_cls=distribution_cls)

    def _generate_mu_grid(self) -> List:
        mu_ranges = self.param_ranges['mu']
        mu_nums = self.param_nums['mu']
        mu_axes = [
            np.linspace(*mu_ranges[dim], mu_nums[dim])
            for dim in sorted(mu_ranges.keys(), key=int)
        ]
        return list(itertools.product(*mu_axes))

    def _generate_cov_grid(self) -> list[np.ndarray]:
        cov_ranges = self.param_ranges['cov']
        cov_nums = self.param_nums['cov']
        keys = sorted(
            (k for k in cov_ranges.keys() if int(k.split('_')[0]) <= int(k.split('_')[1])),
            key=lambda k: (int(k.split('_')[0]), int(k.split('_')[1])),
        )
        indices = [(int(k.split('_')[0]), int(k.split('_')[1])) for k in keys]
        dim = max(max(i, j) for i, j in indices) + 1
        axes = [np.linspace(*cov_ranges[k], cov_nums[k]) for k in keys]
        cov_matrices = []
        for vals in itertools.product(*axes):
            cov = np.zeros((dim, dim))
            for (i, j), v in zip(indices, vals):
                cov[i, j] = v
                if i != j:
                    cov[j, i] = v
            cov_matrices.append(cov)
        return cov_matrices

    def _generate_full_parameter_grid(self) -> Dict:
        mu_grid = self._generate_mu_grid()
        cov_grid = self._generate_cov_grid()
        parameter_grid = {}

        for mu, cov in itertools.product(mu_grid, cov_grid):
            try:
                dist = self.distribution_cls(mu=np.array(mu), cov=np.array(cov))
            except Exception as e:
                print(f"Exception: {e} while initializing the distribution with mu={mu}, cov={cov}.")
                continue

            augmented_eta = dist.augmented_natural_parameters()
            eta = dist.natural_parameters()
            cov_key = tuple(tuple(row) for row in cov)
            parameter_grid[(mu, cov_key)] = {
                "augmented_natural_parameters": augmented_eta,
                "natural_parameters": eta,
                "distribution": dist
            }

        return parameter_grid

    def _generate_corner_points(self) -> List[Dict[str, float]]:
        all_eta = np.stack([v["natural_parameters"] for v in self.parameter_grid.values()])
        eta_min = all_eta.min(axis=0)
        eta_max = all_eta.max(axis=0)
        selected_distributions = [
            v["distribution"]
            for v in self.parameter_grid.values()
            if np.all(
                np.isclose(v["natural_parameters"], eta_min, atol=1e-8) |
                np.isclose(v["natural_parameters"], eta_max, atol=1e-8)
            )
        ]
        return selected_distributions


class OptimizationCornerPointsCompositePrior:
    """
    Corner/grid quadratic-form optimization for a Composite prior.
    The model is expected to accept composite prior parameters via either:
        - model.set_composite_prior_parameters(params, combine_rule=...), or
        - model.set_prior_parameters({"components": params, "combine_rule": ...}, distribution_cls="CompositePrior")
    """

    def __init__(self, posterior_estimator, config: Dict, loss_config: Dict):
        self.posterior_estimator = posterior_estimator
        self.eta_components_cfg: List[Dict[str, Any]] = list(config["eta_components"])

        # Loss lr
        self.loss_lr_corners = loss_config["parameters_box_range"]["ranges"]["lr"]
        self.A_loss, self.b_loss, self.c_loss = self.posterior_estimator.compute_fisher_quadratic_form_lr_only()

        # Prior
        self.A_prior, self.b_prior, self.c_prior = self.posterior_estimator.compute_fisher_quadratic_form_prior_only()

        # Eta grid
        self.eta_corners = self._create_eta_corners()

        # Per-component QFs (for composite prior)
        self.component_names = [cfg.get("name", f"comp{j}") for j, cfg in enumerate(self.eta_components_cfg)]
        theta_blocks = [[j] for j in range(len(self.eta_components_cfg))]
        eta_blocks = [list(range(2 * j, 2 * j + 2)) for j in range(len(self.eta_components_cfg))]
        self.qf_per_component = self.posterior_estimator.compute_prior_only_qf_per_component(
            component_names=self.component_names,
            theta_blocks=theta_blocks,
            eta_blocks=eta_blocks,
        )

    def _evaluate_prior_qf(self, eta_tilde: np.ndarray) -> float:
        return float(eta_tilde @ self.A_prior @ eta_tilde + self.b_prior @ eta_tilde + self.c_prior)

    def _create_eta_corners_through_prior(self, comp_cfg):
        """
        Prefer analytic corners when available, else fall back to grid + hull.
        Returns: [{"params": lambda_dict, "eta": eta_vec}, ...]
        """
        fam = comp_cfg["family"]
        ranges_cfg = comp_cfg["parameters_box_range"]["ranges"]
        ranges_tuples = {k: (float(v[0]), float(v[1])) for k, v in ranges_cfg.items()}
        recs = get_corners(fam, ranges_tuples)
        recs = [{"params": r["params"], "eta": np.asarray(r["eta"], dtype=float).ravel()} for r in recs]

        return recs

    def _create_eta_corners(self):
        """
        Create eta corners
        """
        per_param_corners = []
        for cfg in self.eta_components_cfg:
            e1_lo, e1_hi = cfg["eta_range"]["eta_1"]
            e2_lo, e2_hi = cfg["eta_range"]["eta_2"]
            e1_lo, e1_hi = float(e1_lo), float(e1_hi)
            e2_lo, e2_hi = float(e2_lo), float(e2_hi)

            # 4 combinations per parameter
            per_param_corners.append([
                (e1_lo, e2_lo),
                (e1_lo, e2_hi),
                (e1_hi, e2_lo),
                (e1_hi, e2_hi),
            ])

        # Cartesian product over parameters
        corners = []
        for combo in itertools.product(*per_param_corners):
            flat = [x for pair in combo for x in pair]
            corners.append(np.array(flat))

        return corners

    def evaluate_all_prior_corners(self) -> Tuple:
        results = []

        for eta in tqdm(self.eta_corners, total=len(self.eta_corners), desc="Evaluating corners"):
            est = self._evaluate_prior_qf(eta)
            results.append((eta, est))

        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Largest sensitivity {results[0][1]}.")

        return results, results[0][0]

    def evaluate_all_lr_corners(self) -> List:
        results = []
        for lr in self.loss_lr_corners:
            est = lr**2 * self.A_loss + self.b_loss * lr + self.c_loss
            results.append((lr, est))
            print(f"Lr: {lr} => Estimated FD: {est:.6f}")

        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Lr corner with the largest sensitivity {results[0][1]}: {results[0][0]}.")

        return results

    def _component_qf(self, name: str, eta_j: np.ndarray) -> float:
        """
        Q_j(eta_j) = eta_j^T A_j eta_j + b_j^T eta_j + c_j
        using the principled blockwise QF from PosteriorFDBase.
        """
        A_j, b_j, c_j = self.qf_per_component[name]
        eta_j = np.asarray(eta_j, dtype=float).reshape(-1)
        return float(eta_j @ A_j @ eta_j + b_j @ eta_j + c_j)

    def _component_eta_corners(self, j: int) -> List[np.ndarray]:
        """
        Returns the 4 eta corners for component j as 2D vectors.
        Uses self.eta_components_cfg[j]["eta_range"].
        """
        cfg = self.eta_components_cfg[j]
        e1_lo, e1_hi = cfg["eta_range"]["eta_1"]
        e2_lo, e2_hi = cfg["eta_range"]["eta_2"]
        return [
            np.array([e1_lo, e2_lo], dtype=float),
            np.array([e1_lo, e2_hi], dtype=float),
            np.array([e1_hi, e2_lo], dtype=float),
            np.array([e1_hi, e2_hi], dtype=float),
        ]

    def evaluate_all_prior_corners_per_component(
            self,
            component_names: List[str] = None,
    ) -> Tuple[Dict[str, List[Tuple[np.ndarray, float]]], Dict[str, np.ndarray]]:

        if component_names is None:
            component_names = self.component_names

        results_per_comp: Dict[str, List[Tuple[np.ndarray, float]]] = {}
        eta_star_per_comp: Dict[str, np.ndarray] = {}

        for j, name in enumerate(component_names):
            corners_j = self._component_eta_corners(j)

            vals = []
            for eta_j in corners_j:
                v = self._component_qf(name, eta_j)
                vals.append((eta_j, v))

            vals.sort(key=lambda x: x[1], reverse=True)
            results_per_comp[name] = vals
            eta_star_per_comp[name] = vals[0][0]

        return results_per_comp, eta_star_per_comp

    def _solve_box_qp_2d(self, Ajj: np.ndarray, bj: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        """
        Solve:
            min_x x^T A x + b^T x
            s.t. lo <= x <= hi
        for a 2D variable x.
        """
        Ajj = np.asarray(Ajj, dtype=float)
        bj = np.asarray(bj, dtype=float).reshape(-1)
        lo = np.asarray(lo, dtype=float).reshape(-1)
        hi = np.asarray(hi, dtype=float).reshape(-1)

        x = cp.Variable(2)

        obj = cp.Minimize(cp.quad_form(x, Ajj) + bj @ x)
        constr = [x >= lo, x <= hi]

        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.OSQP, verbose=False)

        return np.array(x.value).reshape(-1)

    def minimize_prior_full_qp(self) -> Tuple[np.ndarray, float]:
        """
        Solve:
            min_{eta in box} eta^T A_prior eta + b_prior^T eta + c_prior
        over the full eta box (all components jointly) using a single QP.

        Returns
        -------
        eta_inf : np.ndarray
            Minimising eta vector.
        val_inf : float
            Minimum QF value.
        """
        dim = len(self.eta_components_cfg) * 2
        lo = np.empty(dim)
        hi = np.empty(dim)
        for j, cfg in enumerate(self.eta_components_cfg):
            lo[2 * j] = float(cfg["eta_range"]["eta_1"][0])
            hi[2 * j] = float(cfg["eta_range"]["eta_1"][1])
            lo[2 * j + 1] = float(cfg["eta_range"]["eta_2"][0])
            hi[2 * j + 1] = float(cfg["eta_range"]["eta_2"][1])

        x = cp.Variable(dim)
        obj = cp.Minimize(cp.quad_form(x, self.A_prior) + self.b_prior @ x)
        prob = cp.Problem(obj, [x >= lo, x <= hi])
        prob.solve(solver=cp.OSQP, verbose=False)

        eta_inf = np.array(x.value, dtype=float).reshape(-1)
        val_inf = float(eta_inf @ self.A_prior @ eta_inf + self.b_prior @ eta_inf + self.c_prior)
        return eta_inf, val_inf

    def minimize_prior_per_component_qp(self, component_names: List[str] = None):

        if component_names is None:
            component_names = self.component_names

        eta_min: Dict[str, np.ndarray] = {}
        values: Dict[str, float] = {}

        for j, name in enumerate(component_names):
            A_j, b_j, c_j = self.qf_per_component[name]
            cfg = self.eta_components_cfg[j]
            lo = np.array([cfg["eta_range"]["eta_1"][0], cfg["eta_range"]["eta_2"][0]], dtype=float)
            hi = np.array([cfg["eta_range"]["eta_1"][1], cfg["eta_range"]["eta_2"][1]], dtype=float)
            eta_star = self._solve_box_qp_2d(A_j, b_j, lo, hi)
            val = float(eta_star @ A_j @ eta_star + b_j @ eta_star + c_j)
            eta_min[name] = eta_star
            values[name] = val

        return eta_min, values

    # -------------------------
    # Black-box optimisation (full box, global)
    # -------------------------
    def _eta_bounds_full_box(self):
        bounds = []
        for cfg in self.eta_components_cfg:
            e1_lo, e1_hi = cfg["eta_range"]["eta_1"]
            e2_lo, e2_hi = cfg["eta_range"]["eta_2"]

            bounds.append((e1_lo, e1_hi))
            bounds.append((e2_lo, e2_hi))

        return bounds

    def _evaluate_prior_fd_black_box(self, eta: np.ndarray) -> float:
        """
        True black-box evaluation of prior-only FD at eta:
            (1/m) sum_i || (s_ref - grad_log_g) - grad_T @ eta ||^2
        This does NOT use A,b,c.

        Requires posterior_estimator to expose:
            fd_prior_only_given_eta(eta) -> float
        """
        eta = np.asarray(eta, dtype=float).reshape(-1)
        return float(self.posterior_estimator.fd_prior_only_given_eta(eta))

    def _run_optimizer(
        self,
        func,
        bounds,
        method: str,
        seed: int,
        maxiter: int,
        popsize: int,
        tol: float,
        polish: bool,
        workers: int,
        updating: str,
        n_restarts: int,
    ):
        """Run sup/inf optimizer with the chosen method, optionally with restarts."""
        _METHODS = ("differential_evolution", "dual_annealing")
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got '{method}'.")

        best_res = None
        best_val = None

        for r in range(max(1, n_restarts)):
            s = seed + r
            if method == "differential_evolution":
                res = differential_evolution(
                    func=func,
                    bounds=bounds,
                    seed=s,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=tol,
                    polish=polish,
                    workers=workers,
                    updating=updating,
                    disp=False,
                )
            else:  # dual_annealing
                res = dual_annealing(
                    func=func,
                    bounds=bounds,
                    seed=s,
                    maxiter=maxiter,
                )

            if best_val is None or res.fun < best_val:
                best_val = res.fun
                best_res = res

        return best_res

    def black_box_optimize_prior_box_global(
        self,
        *,
        method: str = "dual_annealing",
        seed: int = 0,
        maxiter: int = 200,
        popsize: int = 15,
        tol: float = 1e-6,
        polish: bool = True,
        workers: int = 1,
        updating: str = "immediate",
        n_restarts: int = 1,
    ) -> BlackBoxOptResult:
        """
        Solve:
            sup_{eta in Gamma_box} FD(eta)
            inf_{eta in Gamma_box} FD(eta)
        using a global black-box solver.

        Parameters
        ----------
        method : str
            "differential_evolution" or "dual_annealing".
        n_restarts : int
            Number of independent restarts; best result is kept.
        """
        bounds = self._eta_bounds_full_box()

        kwargs = dict(
            method=method, seed=seed, maxiter=maxiter,
            popsize=popsize, tol=tol, polish=polish,
            workers=workers, updating=updating, n_restarts=n_restarts,
        )

        res_sup = self._run_optimizer(func=lambda x: -self._evaluate_prior_fd_black_box(x), bounds=bounds, **kwargs)
        eta_sup = np.asarray(res_sup.x, dtype=float)
        val_sup = float(self._evaluate_prior_fd_black_box(eta_sup))

        res_inf = self._run_optimizer(func=self._evaluate_prior_fd_black_box, bounds=bounds,
                                      **{**kwargs, "seed": seed + n_restarts})
        eta_inf = np.asarray(res_inf.x, dtype=float)
        val_inf = float(self._evaluate_prior_fd_black_box(eta_inf))

        return BlackBoxOptResult(
            eta_sup=eta_sup,
            val_sup=val_sup,
            eta_inf=eta_inf,
            val_inf=val_inf,
            S_hat=float(val_sup - val_inf),
            nfev_sup=int(getattr(res_sup, "nfev", -1)),
            nfev_inf=int(getattr(res_inf, "nfev", -1)),
        )

    def _evaluate_copula_fd_black_box(
        self,
        x: np.ndarray,
        idx_g0: int = 0,
        idx_nu: int = 2,
        apply_z_transform: bool = True,
    ) -> float:
        """
        True black-box evaluation of the FD objective for Gaussian copula sensitivity.

        Parameters
        ----------
        x : np.ndarray
            1D array containing the Gaussian copula correlation parameter.
        idx_g0 : int
            Index of the first component.
        idx_nu : int
            Index of the second component.
        apply_z_transform : bool
            Passed through to fd_gaussian_copula_given_lambda.

        Returns
        -------
        float
            Empirical FD estimate.
        """
        lam = float(np.asarray(x, dtype=float).reshape(-1)[0])
        lower = upper = None
        if apply_z_transform:
            prior = getattr(self.posterior_estimator.model, "prior_init", None)
            if prior is not None and hasattr(prior, "components"):
                lower = np.array([c.low for c in prior.components], dtype=float)
                upper = np.array([c.high for c in prior.components], dtype=float)
        return float(
            self.posterior_estimator.fd_gaussian_copula_given_lambda(
                lam,
                idx_g0=idx_g0,
                idx_nu=idx_nu,
                lower=lower,
                upper=upper,
                apply_z_transform=apply_z_transform,
            )
        )

    def black_box_optimize_gaussian_copula(
        self,
        lambda_range=(-0.95, 0.95),
        seed: int = 0,
        maxiter: int = 200,
        popsize: int = 15,
        tol: float = 1e-6,
        polish: bool = True,
        workers: int = 1,
        updating: str = "immediate",
    ) -> BlackBoxCopulaOptResult:
        """
        Solve:
            sup_{lambda in Gamma} FD_copula(lambda)
            inf_{lambda in Gamma} FD_copula(lambda)

        using a black-box global optimiser.

        If 0 belongs to lambda_range, then the reference prior is in the family
        and the infimum should be attained at lambda = 0 with value 0. We still
        optionally optimise the infimum numerically for consistency with the API.
        """
        lo, hi = map(float, lambda_range)
        bounds = [(lo, hi)]

        # maximise via minimise negative
        res_sup = differential_evolution(
            func=lambda x: -self._evaluate_copula_fd_black_box(x),
            bounds=bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            workers=workers,
            updating=updating,
            disp=False,
        )
        lambda_sup = float(np.asarray(res_sup.x).reshape(-1)[0])
        val_sup = float(self._evaluate_copula_fd_black_box(np.array([lambda_sup])))

        # minimise directly
        res_inf = differential_evolution(
            func=lambda x: self._evaluate_copula_fd_black_box(x),
            bounds=bounds,
            seed=seed + 1,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            workers=workers,
            updating=updating,
            disp=False,
        )
        lambda_inf = float(np.asarray(res_inf.x).reshape(-1)[0])
        val_inf = float(self._evaluate_copula_fd_black_box(np.array([lambda_inf])))

        # if reference value is inside the box, enforce the exact infimum
        if lo <= 0.0 <= hi:
            lambda_inf = 0.0
            val_inf = 0.0

        return BlackBoxCopulaOptResult(
            lambda_sup=lambda_sup,
            val_sup=val_sup,
            lambda_inf=lambda_inf,
            val_inf=val_inf,
            S_hat=float(val_sup - val_inf),
            nfev_sup=int(getattr(res_sup, "nfev", -1)),
            nfev_inf=int(getattr(res_inf, "nfev", -1)),
        )

    def evaluate_gaussian_copula_grid(
        self,
        *,
        lambda_range=(-0.95, 0.0),
        n_grid: int = 101,
        idx_g0: int = 0,
        idx_nu: int = 2,
        apply_z_transform: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Evaluate the Gaussian copula FD objective on a 1D grid.

        Parameters
        ----------
        lambda_range : tuple[float, float]
            Range of Gaussian copula correlation parameter.
        n_grid : int
            Number of grid points.
        idx_g0 : int
            Index of the first component.
        idx_nu : int
            Index of the second component.
        apply_z_transform : bool
            If True (default), rescale samples to (0,1) and apply Phi^{-1}.
            If False, samples are used directly as z-scores.

        Returns
        -------
        List[Tuple[float, float]]
            Pairs (lambda, FD(lambda)), sorted by lambda.
        """
        lo, hi = map(float, lambda_range)
        grid = np.linspace(lo, hi, int(n_grid))

        results = []
        for lam in grid:
            val = self._evaluate_copula_fd_black_box(
                np.array([lam], dtype=float),
                idx_g0=idx_g0,
                idx_nu=idx_nu,
                apply_z_transform=apply_z_transform,
            )
            results.append((float(lam), float(val)))

        return results

    def evaluate_gaussian_copula_grid_and_argmax(
        self,
        *,
        lambda_range=(-0.95, 0.0),
        n_grid: int = 101,
        idx_g0: int = 0,
        idx_nu: int = 2,
        apply_z_transform: bool = True,
    ) -> Tuple[List[Tuple[float, float]], float, float]:
        """
        Evaluate the Gaussian copula FD objective on a grid and return
        the maximiser on that grid.

        Parameters
        ----------
        lambda_range : tuple[float, float]
            Range of Gaussian copula correlation parameter.
        n_grid : int
            Number of grid points.
        idx_g0 : int
            Index of the first component.
        idx_nu : int
            Index of the second component.
        apply_z_transform : bool
            If True (default), rescale samples to (0,1) and apply Phi^{-1}.
            If False, samples are used directly as z-scores.

        Returns
        -------
        results : list of (lambda, value)
        lambda_star : float
            Grid maximiser.
        val_star : float
            Maximum FD value on the grid.
        """
        results = self.evaluate_gaussian_copula_grid(
            lambda_range=lambda_range,
            n_grid=n_grid,
            idx_g0=idx_g0,
            idx_nu=idx_nu,
            apply_z_transform=apply_z_transform,
        )
        lambda_star, val_star = max(results, key=lambda x: x[1])
        return results, float(lambda_star), float(val_star)

    def _evaluate_fgm_copula_fd_black_box(
            self,
            x: np.ndarray,
            idx_g0: int = 0,
            idx_nu: int = 2,
    ) -> float:
        lam = float(np.asarray(x, dtype=float).reshape(-1)[0])
        return float(
            self.posterior_estimator.fd_fgm_copula_given_lambda(
                lam,
                idx_g0=idx_g0,
                idx_nu=idx_nu,
            )
        )

    def black_box_optimize_fgm_copula(
        self,
        lambda_range=(-1.0, 1.0),
        seed: int = 0,
        maxiter: int = 200,
        popsize: int = 15,
        tol: float = 1e-6,
        polish: bool = True,
        workers: int = 1,
        updating: str = "immediate",
    ) -> BlackBoxCopulaOptResult:
        """
        Solve:
            sup_{lambda in Gamma} FD_fgm(lambda)
            inf_{lambda in Gamma} FD_fgm(lambda)

        using a black-box global optimiser (differential evolution).
        """
        lo, hi = map(float, lambda_range)
        bounds = [(lo, hi)]

        res_sup = differential_evolution(
            func=lambda x: -self._evaluate_fgm_copula_fd_black_box(x),
            bounds=bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            workers=workers,
            updating=updating,
            disp=False,
        )
        lambda_sup = float(np.asarray(res_sup.x).reshape(-1)[0])
        val_sup = float(self._evaluate_fgm_copula_fd_black_box(np.array([lambda_sup])))

        res_inf = differential_evolution(
            func=lambda x: self._evaluate_fgm_copula_fd_black_box(x),
            bounds=bounds,
            seed=seed + 1,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            workers=workers,
            updating=updating,
            disp=False,
        )
        lambda_inf = float(np.asarray(res_inf.x).reshape(-1)[0])
        val_inf = float(self._evaluate_fgm_copula_fd_black_box(np.array([lambda_inf])))

        if lo <= 0.0 <= hi:
            lambda_inf = 0.0
            val_inf = 0.0

        return BlackBoxCopulaOptResult(
            lambda_sup=lambda_sup,
            val_sup=val_sup,
            lambda_inf=lambda_inf,
            val_inf=val_inf,
            S_hat=float(val_sup - val_inf),
            nfev_sup=int(getattr(res_sup, "nfev", -1)),
            nfev_inf=int(getattr(res_inf, "nfev", -1)),
        )

    def evaluate_fgm_copula_grid(
        self,
        *,
        lambda_range=(-1.0, 1.0),
        n_grid: int = 101,
        idx_g0: int = 0,
        idx_nu: int = 2,
    ) -> List[Tuple[float, float]]:
        """
        Evaluate the FGM copula FD objective on a 1D grid.

        Returns
        -------
        List[Tuple[float, float]]
            Pairs (lambda, FD(lambda)), sorted by lambda.
        """
        lo, hi = map(float, lambda_range)
        grid = np.linspace(lo, hi, int(n_grid))

        results = []
        for lam in grid:
            val = self._evaluate_fgm_copula_fd_black_box(
                np.array([lam], dtype=float),
                idx_g0=idx_g0,
                idx_nu=idx_nu,
            )
            results.append((float(lam), float(val)))

        return results

    def evaluate_fgm_copula_grid_and_argmax(
        self,
        *,
        lambda_range=(-1.0, 1.0),
        n_grid: int = 101,
        idx_g0: int = 0,
        idx_nu: int = 2,
    ) -> Tuple[List[Tuple[float, float]], float, float]:
        """
        Evaluate the FGM copula FD objective on a grid and return the maximiser.

        Returns
        -------
        results : list of (lambda, value)
        lambda_star : float
        val_star : float
        """
        results = self.evaluate_fgm_copula_grid(
            lambda_range=lambda_range,
            n_grid=n_grid,
            idx_g0=idx_g0,
            idx_nu=idx_nu,
        )
        lambda_star, val_star = max(results, key=lambda x: x[1])
        return results, float(lambda_star), float(val_star)

    def _evaluate_frank_copula_fd_black_box(
            self,
            x: np.ndarray,
            idx_g0: int = 0,
            idx_nu: int = 2,
    ) -> float:
        lam = float(np.asarray(x, dtype=float).reshape(-1)[0])
        return float(
            self.posterior_estimator.fd_frank_copula_given_lambda(
                lam,
                idx_g0=idx_g0,
                idx_nu=idx_nu,
            )
        )

    def black_box_optimize_frank_copula(
        self,
        lambda_range=(-0.1, 0.1),
        seed: int = 0,
        maxiter: int = 200,
        popsize: int = 15,
        tol: float = 1e-6,
        polish: bool = True,
        workers: int = 1,
        updating: str = "immediate",
    ) -> BlackBoxCopulaOptResult:
        """
        Solve:
            sup_{theta in Gamma} FD_frank(theta)
            inf_{theta in Gamma} FD_frank(theta)

        using a black-box global optimiser (differential evolution).
        FD is extended by continuity: FD(0) = 0.
        """
        lo, hi = map(float, lambda_range)
        bounds = [(lo, hi)]

        res_sup = differential_evolution(
            func=lambda x: -self._evaluate_frank_copula_fd_black_box(x),
            bounds=bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            workers=workers,
            updating=updating,
            disp=False,
        )
        lambda_sup = float(np.asarray(res_sup.x).reshape(-1)[0])
        val_sup = float(self._evaluate_frank_copula_fd_black_box(np.array([lambda_sup])))

        res_inf = differential_evolution(
            func=lambda x: self._evaluate_frank_copula_fd_black_box(x),
            bounds=bounds,
            seed=seed + 1,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            workers=workers,
            updating=updating,
            disp=False,
        )
        lambda_inf = float(np.asarray(res_inf.x).reshape(-1)[0])
        val_inf = float(self._evaluate_frank_copula_fd_black_box(np.array([lambda_inf])))

        # θ=0 gives FD=0 (independence); enforce exact infimum if 0 is in range
        if lo <= 0.0 <= hi:
            lambda_inf = 0.0
            val_inf = 0.0

        return BlackBoxCopulaOptResult(
            lambda_sup=lambda_sup,
            val_sup=val_sup,
            lambda_inf=lambda_inf,
            val_inf=val_inf,
            S_hat=float(val_sup - val_inf),
            nfev_sup=int(getattr(res_sup, "nfev", -1)),
            nfev_inf=int(getattr(res_inf, "nfev", -1)),
        )

    def evaluate_frank_copula_grid(
        self,
        *,
        lambda_range=(-0.1, 0.1),
        n_grid: int = 101,
        idx_g0: int = 0,
        idx_nu: int = 2,
    ) -> List[Tuple[float, float]]:
        """
        Evaluate the Frank copula FD objective on a 1D grid.

        Returns
        -------
        List[Tuple[float, float]]
            Pairs (theta, FD(theta)), sorted by theta.
        """
        lo, hi = map(float, lambda_range)
        grid = np.linspace(lo, hi, int(n_grid))

        results = []
        for lam in grid:
            val = self._evaluate_frank_copula_fd_black_box(
                np.array([lam], dtype=float),
                idx_g0=idx_g0,
                idx_nu=idx_nu,
            )
            results.append((float(lam), float(val)))

        return results

    def evaluate_frank_copula_grid_and_argmax(
        self,
        *,
        lambda_range=(-0.1, 0.1),
        n_grid: int = 101,
        idx_g0: int = 0,
        idx_nu: int = 2,
    ) -> Tuple[List[Tuple[float, float]], float, float]:
        """
        Evaluate the Frank copula FD objective on a grid and return the maximiser.

        Returns
        -------
        results : list of (theta, value)
        lambda_star : float
        val_star : float
        """
        results = self.evaluate_frank_copula_grid(
            lambda_range=lambda_range,
            n_grid=n_grid,
            idx_g0=idx_g0,
            idx_nu=idx_nu,
        )
        lambda_star, val_star = max(results, key=lambda x: x[1])
        return results, float(lambda_star), float(val_star)


class OptimizationCornerPointsGamma(OptimizationCornerPointsBase):
    def __init__(
        self,
        posterior_estimator,
        prior_config: Dict,
        loss_config: Dict,
        distribution_cls=Gamma,
    ):
        """
        Grid/corner quadratic form generation and optimization for gamma.
        """
        super().__init__(posterior_estimator=posterior_estimator, prior_config=prior_config,
                         loss_config=loss_config, distribution_cls=distribution_cls)

    def _generate_alpha_grid(self) -> List:
        alpha_ranges = self.param_ranges['alpha']
        alpha_num = self.param_nums['alpha']
        alpha_grid = np.linspace(alpha_ranges[0], alpha_ranges[-1], alpha_num).tolist()

        return alpha_grid

    def _generate_theta_grid(self) -> List[np.ndarray]:
        theta_ranges = self.param_ranges['theta']
        theta_num = self.param_nums['theta']
        theta_grid = np.linspace(theta_ranges[0], theta_ranges[-1], theta_num).tolist()

        return theta_grid

    def _generate_full_parameter_grid(self) -> Dict:
        alpha_grid = self._generate_alpha_grid()
        theta_grid = self._generate_theta_grid()
        parameter_grid = {}

        for alpha, theta in itertools.product(alpha_grid, theta_grid):
            try:
                dist = self.distribution_cls(alpha=alpha, theta=theta)
            except Exception as e:
                print(f"Exception: {e} while initializing the distribution with alpha={alpha}, theta={theta}.")
                continue

            augmented_eta = dist.augmented_natural_parameters()
            eta = dist.natural_parameters()
            parameter_grid[(alpha, theta)] = {
                "augmented_natural_parameters": augmented_eta,
                "natural_parameters": eta,
                "distribution": dist
            }
        return parameter_grid


class OptimizationCornerPointsInverseWishart(OptimizationCornerPointsBase):
    def __init__(
        self,
        posterior_estimator,
        prior_config: Dict,
        loss_config: Dict,
        distribution_cls=InverseWishart,
    ):
        """
        Grid/corner quadratic form generation and optimization for Inverse Wishart.
        """
        super().__init__(posterior_estimator=posterior_estimator, prior_config=prior_config,
                         loss_config=loss_config, distribution_cls=distribution_cls)

    def _generate_df_grid(self) -> np.ndarray:
        return np.linspace(self.param_ranges["df"][0], self.param_ranges["df"][1], self.param_nums["df"])

    def _generate_scale_grid(self) -> List[np.ndarray]:
        scale_ranges = self.param_ranges['scale']
        scale_nums = self.param_nums['scale']
        keys = sorted(scale_ranges.keys())
        axes = [
            np.linspace(*scale_ranges[k], scale_nums[k]) for k in keys
        ]
        scale_matrices = []
        for values in itertools.product(*axes):
            scale = np.zeros((2, 2))  # assumes 2D, generalize if needed
            for idx, k in enumerate(keys):
                i, j = map(int, k.split('_'))
                scale[i, j] = values[idx]
            scale_matrices.append(scale)
        return scale_matrices

    def _generate_full_parameter_grid(self) -> Dict:
        df_grid = self._generate_df_grid()
        scale_grid = self._generate_scale_grid()
        parameter_grid = {}

        for df, scale in itertools.product(df_grid, scale_grid):
            try:
                dist = self.distribution_cls(df=df, scale=np.array(scale))
            except Exception as e:
                print(f"Exception: {e} while initializing the distribution with df={df}, scale={scale}.")
                continue

            augmented_eta = dist.augmented_natural_parameters()
            eta = dist.natural_parameters()
            scale_key = tuple(tuple(row) for row in scale)
            parameter_grid[(df, scale_key)] = {
                "augmented_natural_parameters": augmented_eta,
                "natural_parameters": eta,
                "distribution": dist
            }

        return parameter_grid

    def _generate_corner_points(self) -> List[Dict[str, float]]:
        all_eta = np.stack([v["natural_parameters"] for v in self.parameter_grid.values()])
        eta_min = all_eta.min(axis=0)
        eta_max = all_eta.max(axis=0)
        corners = list(itertools.product(*zip(eta_min, eta_max)))
        corner_set = {tuple(np.round(corner, 8)) for corner in corners}
        selected_distributions = [
            v["distribution"]
            for v in self.parameter_grid.values()
            if tuple(np.round(v["natural_parameters"], 8)) in corner_set
        ]

        return selected_distributions
