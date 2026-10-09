from typing import Dict, List, Tuple, Any
import itertools
import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull
from tqdm import tqdm
from scipy.optimize import differential_evolution, dual_annealing
from dataclasses import dataclass

from src.common.distributions.gaussian import Gaussian, MultivariateGaussian


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

        self.Lambda_prior, self.b_prior, self.c_prior = self.posterior_estimator.compute_fisher_quadratic_form_prior_only()

        self.parameter_grid = self._generate_full_parameter_grid()
        self.distribution_corner_points: List[Dict[str, float]] = self._generate_corner_points()

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
    """Corner/grid quadratic-form and black-box FD optimisation for a composite prior."""

    def __init__(self, posterior_estimator, config: Dict, loss_config: Dict):
        self.posterior_estimator = posterior_estimator
        self.eta_components_cfg: List[Dict[str, Any]] = list(config["eta_components"])

        # Prior
        self.A_prior, self.b_prior, self.c_prior = self.posterior_estimator.compute_fisher_quadratic_form_prior_only()

        # Eta grid (4^d corners), built on first use
        self._eta_corners = None

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

    @property
    def eta_corners(self):
        if self._eta_corners is None:
            self._eta_corners = self._create_eta_corners()
        return self._eta_corners

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

    def _component_qf(self, name: str, eta_j: np.ndarray) -> float:
        """Evaluate the quadratic form eta_j^T A_j eta_j + b_j^T eta_j + c_j of component name."""
        A_j, b_j, c_j = self.qf_per_component[name]
        eta_j = np.asarray(eta_j, dtype=float).reshape(-1)
        return float(eta_j @ A_j @ eta_j + b_j @ eta_j + c_j)

    def _component_eta_corners(self, j: int) -> List[np.ndarray]:
        """Return the four 2D eta corners of component j from its configured eta_range."""
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
        """Solve the 2D box-constrained QP min x^T A x + b^T x subject to lo <= x <= hi."""
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
        """Minimise the full prior quadratic form jointly over the eta box and return (eta_inf, val_inf)."""
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
        """Evaluate the prior-only FD at eta directly via the posterior estimator, without the QF."""
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
        compute_inf: bool = True,
    ) -> BlackBoxOptResult:
        """Find the sup and inf of the prior-only FD over the eta box with a global black-box solver.

        With compute_inf=False only the sup is searched (e.g. when the inf is known to be 0); eta_inf is then NaN.
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

        if compute_inf:
            res_inf = self._run_optimizer(func=self._evaluate_prior_fd_black_box, bounds=bounds,
                                          **{**kwargs, "seed": seed + n_restarts})
            eta_inf = np.asarray(res_inf.x, dtype=float)
            val_inf = float(self._evaluate_prior_fd_black_box(eta_inf))
        else:
            res_inf, eta_inf, val_inf = None, np.full(len(bounds), np.nan), 0.0

        return BlackBoxOptResult(
            eta_sup=eta_sup,
            val_sup=val_sup,
            eta_inf=eta_inf,
            val_inf=val_inf,
            S_hat=float(val_sup - val_inf),
            nfev_sup=int(getattr(res_sup, "nfev", -1)),
            nfev_inf=int(getattr(res_inf, "nfev", -1)) if res_inf is not None else 0,
        )

    def _evaluate_copula_fd_black_box(
        self,
        x: np.ndarray,
        idx_g0: int = 0,
        idx_nu: int = 2,
        apply_z_transform: bool = True,
    ) -> float:
        """Evaluate the empirical Gaussian-copula FD at the correlation parameter in x."""
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
        """Find the sup and inf of the Gaussian-copula FD over lambda_range with differential evolution."""
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
        """Evaluate the Gaussian-copula FD on a 1D lambda grid and return (lambda, FD) pairs."""
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
        """Evaluate the Gaussian-copula FD on a lambda grid and return the results, argmax and maximum."""
        results = self.evaluate_gaussian_copula_grid(
            lambda_range=lambda_range,
            n_grid=n_grid,
            idx_g0=idx_g0,
            idx_nu=idx_nu,
            apply_z_transform=apply_z_transform,
        )
        lambda_star, val_star = max(results, key=lambda x: x[1])
        return results, float(lambda_star), float(val_star)
