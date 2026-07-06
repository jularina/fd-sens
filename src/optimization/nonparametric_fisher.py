from typing import Dict
import numpy as np
import cvxpy as cp
from omegaconf import OmegaConf

from src.utils.basis_functions import BASIS_FUNCTIONS_REGISTRY


class OptimisationNonparametricBase:
    def __init__(
        self,
        posterior_estimator,
        prior_estimator,
        config: Dict,
        radius: float = 0.0,
        add_nuggets: bool = False,
    ):
        """
        Base class to handle nonparametric quadratic form optimization.
        """
        self.posterior_estimator = posterior_estimator
        self.prior_estimator = prior_estimator

        basis_cls_name = config["basis_funcs_type"]
        basis_cls = BASIS_FUNCTIONS_REGISTRY[basis_cls_name]
        basis_kwargs = config.get("basis_funcs_kwargs", {})
        basis_kwargs = OmegaConf.to_container(basis_kwargs, resolve=True)
        basis_kwargs["prior_samples"] = self.prior_estimator.samples
        basis_kwargs["posterior_samples"] = self.posterior_estimator.samples
        self.basis_function = basis_cls(**basis_kwargs)

        self.A, self.b, self.c = self.posterior_estimator.compute_non_parametric_fisher_quadratic_form_prior_only(
            self.basis_function,
        )
        self.A_c, self.b_c, self.c_c = self.prior_estimator.compute_non_parametric_fisher_quadratic_form_prior_only(
            self.basis_function,
        )
        self.A = self._sym(self.A)
        self.A_c = self._sym(self.A_c)
        psd, rank = self._check_pd_psd_and_rank(self.A)
        print(f"A is {psd} with rank {rank}.")
        psd, rank = self._check_pd_psd_and_rank(self.A_c)
        print(f"A_c is {psd} with rank {rank}.")

        self.d = self.A_c.shape[0]

        if add_nuggets:
            avg_prior = np.trace(self.A_c) / max(self.d, 1)
            avg_obj = np.trace(self.A) / max(self.d, 1)
            eps_prior = max(1e-8, 1e-2 * avg_prior)
            eps_obj = max(1e-12, 1e-4 * avg_obj)
            self.A_c = self._sym(self.A_c + eps_prior * np.eye(self.d))
            self.A = self._sym(self.A + eps_obj * np.eye(self.d))
            psd, rank = self._check_pd_psd_and_rank(self.A)
            print(f"After nuggets A is {psd} with rank {rank}.")
            psd, rank = self._check_pd_psd_and_rank(self.A_c)
            print(f"After nuggets A_c is {psd} with rank {rank}.")

        self.r = self._compute_min_radius() + radius
        print(f"Radius {self.r}.")

    @staticmethod
    def _solve_psd(A: np.ndarray, b: np.ndarray):
        """
        Tries to solve A x = b with a PSD/PD matrix A:
        - First attempt: Cholesky (fast, PD).
        - Fallback: lstsq (robust), equivalent to (near)pinv multiply.
        Returns x and a flag whether PD/cholesky succeeded.
        """
        try:
            L = np.linalg.cholesky(A)
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
            return x, True
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            return x, False

    def _check_pd_psd_and_rank(
            self,
            A: np.ndarray,
            tol: float = 1e-10,
    ):
        """
        Check whether a symmetric matrix is positive definite (PD),
        positive semidefinite (PSD), or not PSD, and return its numerical rank.
        """
        eigvals = np.linalg.eigvalsh(A)
        rank = int(np.sum(eigvals > tol))

        if np.all(eigvals > tol):
            status = "PD"
        elif np.all(eigvals >= -tol):
            status = "PSD"
        else:
            status = "NOT_PSD"

        return status, rank

    @staticmethod
    def _pinv_psd(A: np.ndarray, rcond: float | None = None) -> np.ndarray:
        """
        Moore–Penrose pseudoinverse specialized for symmetric PSD/Hermitian:
        eigen-decompose, invert eigenvalues above tolerance.
        """
        w, V = np.linalg.eigh(A)

        if rcond is None:
            rcond = np.finfo(A.dtype).eps * max(A.shape) * max(w.max(), 1.0)
        w_inv = np.zeros_like(w)
        mask = w > rcond
        w_inv[mask] = 1.0 / w[mask]
        return (V * w_inv) @ V.T

    def _evaluate_qf(self, lam: np.ndarray) -> float:
        return lam @ self.A @ lam + self.b @ lam + self.c

    def _evaluate_constraint_qf(self, lam: np.ndarray) -> float:
        return lam @ self.A_c @ lam + self.b_c @ lam + self.c_c

    def _compute_min_radius(self) -> float:
        quad_term = -0.25 * float(self.b_c.T @ self._pinv_psd(self.A_c) @ self.b_c)
        min_val = float(self.c_c + quad_term)
        print(f"Computed min radius threshold: {min_val}.")
        return max(0.0, min_val)

    # -------------------------
    # Helpers for dual methods
    # -------------------------

    def _M_s(self, omega: float):
        M = omega * self.A_c - self.A
        s = omega * self.b_c - self.b
        return M, s

    def _lam_from_omega(self, omega: float, tol: float) -> np.ndarray:
        M, s = self._M_s(omega)
        M_pinv = np.linalg.pinv(M, rcond=tol)
        return -0.5 * (M_pinv @ s)

    def _constraint(self, lam: np.ndarray) -> float:
        return float(lam.T @ self.A_c @ lam + self.b_c.T @ lam + self.c_c)

    def _objective(self, lam: np.ndarray) -> float:
        return float(lam.T @ self.A @ lam + self.b.T @ lam + self.c)

    def _max_eig_sym(self, X: np.ndarray) -> float:
        Xs = 0.5 * (X + X.T)
        return float(np.linalg.eigvalsh(Xs).max())

    def _range_ok(self, M: np.ndarray, s: np.ndarray, tol: float) -> bool:
        M_pinv = np.linalg.pinv(M, rcond=tol)
        resid = s - M @ (M_pinv @ s)
        return float(np.linalg.norm(resid)) <= 1e3 * tol * (1.0 + float(np.linalg.norm(s)))

    def _dual_1d_value_strict(self, omega: float, radius: float, tol: float) -> float:
        """
        Dual objective d(omega) = c - omega*(c_c - r) - 1/4 s^T M^† s.
        Returns +inf if domain conditions fail.
        """
        if omega < 0:
            return float("inf")

        r = float(radius)
        M, s = self._M_s(omega)

        if self._max_eig_sym(M) > 1e3 * tol:
            return float("inf")
        if not self._range_ok(M, s, tol):
            return float("inf")

        M_pinv = np.linalg.pinv(M, rcond=tol)
        quad = float(s.T @ (M_pinv @ s))
        return float(self.c - omega * (self.c_c - r) - 0.25 * quad)

    def _bracket_feasible_omega(self, radius: float, tol: float, omega_max: float, grid: int):
        """
        Find an interval [L,U] that contains feasible values of the dual variable omega.
        """
        xs = np.linspace(0.0, omega_max, grid)
        feas = []
        for omega in xs:
            val = self._dual_1d_value_strict(omega, radius, tol)
            feas.append(np.isfinite(val))
        feas = np.array(feas, dtype=bool)
        if not feas.any():
            return None

        idx = np.where(feas)[0]
        start = idx[0]
        end = start
        for j in idx[1:]:
            if j == end + 1:
                end = j
            else:
                break
        L = float(xs[start])
        U = float(xs[end])

        if start == end:
            if start > 0:
                L = float(xs[start - 1])
            if start < len(xs) - 1:
                U = float(xs[start + 1])
        return (max(0.0, L), max(0.0, U))

    def _refine_omega_by_active_constraint(
        self,
        omega_init: float,
        radius: float,
        tol: float,
        omega_lo: float,
        omega_hi: float,
        max_iter: int = 80,
    ) -> float:
        """
        If omega>0, KKT suggests g(lambda(omega)) = r.
        Enforce by bisection on phi(omega) = g(lambda(omega)) - r over a feasible bracket.
        """
        r = float(radius)

        def phi(omega: float) -> float:
            lam = self._lam_from_omega(omega, tol)
            return self._constraint(lam) - r

        plo = phi(omega_lo)
        phi_ = phi(omega_hi)

        if not np.isfinite(plo) or not np.isfinite(phi_) or plo * phi_ > 0:
            return float(omega_init)

        a, b = omega_lo, omega_hi
        fa, fb = plo, phi_

        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = phi(m)
            if not np.isfinite(fm):
                break
            if abs(fm) <= 1e2 * tol:
                return float(m)
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm

        return float(0.5 * (a + b))

    def is_singular_symmetric(self, M, tol=1e-10):
        eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
        return np.min(np.abs(eigvals)) <= tol

    @staticmethod
    def _sym(A: np.ndarray) -> np.ndarray:
        return 0.5 * (A + A.T)

    @staticmethod
    def _canonical_sign(lam: np.ndarray) -> np.ndarray:
        """Make the largest-magnitude component positive to break sign ambiguity."""
        return lam if lam[np.argmax(np.abs(lam))] >= 0 else -lam

    def _complete_lambda_from_kernel(
        self, lam_p: np.ndarray, M: np.ndarray, tol: float = 1e-8
    ) -> np.ndarray:
        """
        When s(omega*) = 0, the KKT particular solution lam_p = 0.
        The optimum lies in ker(M(omega*)): pick the null eigenvector that
        maximises the objective and scale it to satisfy the active constraint
        lambda^T A_c lambda = r.
        """
        M_sym = self._sym(M)
        w, V = np.linalg.eigh(M_sym)
        null_tol = tol * max(float(np.abs(w).max()), 1.0) * 1e3
        null_mask = np.abs(w) <= null_tol
        if not np.any(null_mask):
            return lam_p

        null_vecs = V[:, null_mask]
        objectives = np.array([float(v @ self.A @ v) for v in null_vecs.T])
        v = null_vecs[:, np.argmax(np.abs(objectives))]

        v_Ac_v = float(v @ self.A_c @ v)
        if v_Ac_v <= tol:
            return lam_p

        alpha = np.sqrt(max(self.r, 0.0) / v_Ac_v)
        lam_pos = lam_p + alpha * v
        lam_neg = lam_p - alpha * v
        return lam_pos if self._evaluate_qf(lam_pos) >= self._evaluate_qf(lam_neg) else lam_neg

    # -------------------------
    # Optimisation methods
    # -------------------------

    def optimize_through_sdp_relaxation(self):
        """
        Primal SDP relaxation: lift lambda*lambda^T to H and relax H = lambda*lambda^T
        to [[H, lambda], [lambda^T, 1]] >= 0 (Schur complement form).

        When b = b_c = 0, lambda does not appear in the objective or constraint,
        so the solver sets it to 0. In that case lambda is recovered from the
        leading eigenvector of H (rank-1 extraction).
        """
        lam = cp.Variable(self.d)
        H = cp.Variable((self.d, self.d), symmetric=True)

        objective = cp.Maximize(cp.trace(self.A @ H) + self.b @ lam + self.c)
        constraint1 = (
            cp.trace(self.A_c @ H) + self.b_c @ lam + self.c_c <= self.r
        )
        schur_matrix = cp.bmat([
            [H,  cp.reshape(lam, (self.d, 1), order='C')],
            [cp.reshape(lam, (1, self.d), order='C'), cp.Constant([[1]])]
        ])
        constraint2 = schur_matrix >> 0

        print("cond(A):", np.linalg.cond(self.A))
        print("cond(A_c):", np.linalg.cond(self.A_c))
        print("r:", self.r)

        problem = cp.Problem(objective, [constraint1, constraint2])
        problem.solve(solver=cp.MOSEK)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"SDP optimization failed: {problem.status}")

        lam_val = lam.value
        H_val = H.value

        H_rank, _ = self._check_pd_psd_and_rank(H_val)
        H_numerical_rank = int(np.sum(np.linalg.eigvalsh(H_val) > 1e-6 * max(np.linalg.eigvalsh(H_val).max(), 1.0)))
        print(f"H is rank-1: {H_numerical_rank == 1} (numerical rank: {H_numerical_rank}, status: {H_rank})")

        # When b = b_c = 0, lambda does not appear in the objective or constraint
        # so the solver leaves it at 0. Recover lambda from the leading eigenvector
        # of H (rank-1 extraction: H = lambda*lambda^T at the optimum).
        if np.linalg.norm(lam_val) < 1e-8 * max(np.linalg.norm(H_val), 1.0):
            w, V = np.linalg.eigh(H_val)
            lam_val = V[:, -1] * np.sqrt(max(w[-1], 0.0))
        lam_val = self._canonical_sign(lam_val)

        primal_value = self._evaluate_qf(lam_val)
        constraint_value = self._evaluate_constraint_qf(lam_val)

        return {
            "lambda_star": lam_val,
            "H_opt": H_val,
            "primal_value": primal_value,
            "constraint_value": constraint_value,
            "dual_value": problem.value,
        }

    def optimize_through_dual_1d_lambda(
            self,
            tol: float = 1e-10,
            omega_init: float | None = None,
            omega_cap: float = 1e12,
            max_expand: int = 80,
            max_iter: int = 200,
    ):
        """
        Solve   max_{lambda}  lambda^T A lambda + b^T lambda + c
                s.t.  lambda^T A_c lambda + b_c^T lambda + c_c <= r

        via the 1D dual in omega (the Lagrange multiplier):
            inf_{omega >= 0}  d(omega) = c - omega*(c_c - r) + 1/4 s(omega)^T M(omega)^† s(omega)
        where M(omega) = omega*A_c - A,  s(omega) = omega*b_c - b.
        """
        A = self._sym(self.A)
        Ac = self._sym(self.A_c)
        b = np.asarray(self.b).reshape(-1)
        bc = np.asarray(self.b_c).reshape(-1)
        c0 = float(self.c)
        cc = float(self.c_c)
        r = float(self.r)
        n = A.shape[0]

        def eval_dual_and_lam(omega: float):
            """
            Returns (feasible: bool, dval: float, lam: np.ndarray | None).
            """
            if omega < 0.0:
                return False, float("inf"), None

            M = self._sym(omega * Ac - A)
            s = omega * bc - b

            w, V = np.linalg.eigh(M)
            wmax = float(w.max()) if w.size else 0.0

            eps = 1e3 * tol * max(1.0, abs(wmax))
            if float(w.min()) < -eps:
                return False, float("inf"), None

            null_mask = w <= eps
            if np.any(null_mask):
                U0 = V[:, null_mask]
                proj = U0.T @ s
                if float(np.linalg.norm(proj)) > 1e3 * tol * (1.0 + float(np.linalg.norm(s))):
                    return False, float("inf"), None

            w_inv = np.zeros_like(w)
            pos_mask = w > eps
            w_inv[pos_mask] = 1.0 / w[pos_mask]
            M_pinv = (V * w_inv) @ V.T

            quad = float(s @ (M_pinv @ s))
            dval = float(c0 - omega * (cc - r) + 0.25 * quad)

            lam_vec = -0.5 * (M_pinv @ s)
            # When s = 0, lam_vec = 0; recover from ker(M) instead.
            if np.linalg.norm(s) <= tol * max(float(np.linalg.norm(bc)), 1.0):
                lam_vec = self._complete_lambda_from_kernel(lam_vec, M, tol)
            lam_vec = self._canonical_sign(lam_vec)
            return True, dval, lam_vec

        def find_omega_min_feasible():
            """
            Find the smallest feasible omega by exponential search then bisection.
            """
            if omega_init is None:
                omega_hi = 1.0
            else:
                omega_hi = max(0.0, float(omega_init))

            feas, _, _ = eval_dual_and_lam(omega_hi)
            expand = 0
            while not feas and omega_hi < omega_cap and expand < max_expand:
                omega_hi = 2.0 * omega_hi if omega_hi > 0 else 1.0
                feas, _, _ = eval_dual_and_lam(omega_hi)
                expand += 1

            if not feas:
                return None

            omega_lo = 0.0
            for _ in range(max_iter):
                mid = 0.5 * (omega_lo + omega_hi)
                feas_mid, _, _ = eval_dual_and_lam(mid)
                if feas_mid:
                    omega_hi = mid
                else:
                    omega_lo = mid
                if abs(omega_hi - omega_lo) <= 1e-14 * (1.0 + abs(omega_hi) + abs(omega_lo)):
                    break
            return float(omega_hi)

        def bracket_minimizer(omega0: float):
            """
            Produce a bracket [pt_a, pt_b, pt_c] with pt_a < pt_b < pt_c
            and f(pt_b) <= f(pt_a), f(pt_b) <= f(pt_c).
            """
            pt_a = omega0
            fa_ok, fa, _ = eval_dual_and_lam(pt_a)
            if not fa_ok:
                return None

            step = max(1e-6, 0.05 * (1.0 + pt_a))
            pt_b = pt_a + step
            fb_ok, fb, _ = eval_dual_and_lam(pt_b)

            tries = 0
            while (not fb_ok) and pt_b < omega_cap and tries < max_expand:
                step *= 2.0
                pt_b = pt_a + step
                fb_ok, fb, _ = eval_dual_and_lam(pt_b)
                tries += 1
            if not fb_ok:
                return None

            if fb >= fa:
                return (pt_a, pt_a, pt_b, fa, fa, fb)

            pt_c = pt_b + step
            fc_ok, fc, _ = eval_dual_and_lam(pt_c)
            tries = 0
            while (not fc_ok or fc < fb) and pt_c < omega_cap and tries < max_expand:
                step *= 2.0
                pt_a, fa = pt_b, fb
                pt_b, fb = pt_c, fc if fc_ok else fb
                pt_c = pt_b + step
                fc_ok, fc, _ = eval_dual_and_lam(pt_c)
                tries += 1

            if not fc_ok:
                return None

            return (pt_a, pt_b, pt_c, fa, fb, fc)

        def golden_section(a: float, b: float, c: float):
            """
            Minimize on [a, c] assuming unimodal/convex in omega.
            """
            if b == a:
                return a

            left, right = a, c
            gr = 0.5 * (np.sqrt(5.0) - 1.0)

            x1 = right - gr * (right - left)
            x2 = left + gr * (right - left)

            f1_ok, f1, _ = eval_dual_and_lam(x1)
            f2_ok, f2, _ = eval_dual_and_lam(x2)

            for _ in range(max_iter):
                if (right - left) <= 1e-12 * (1.0 + abs(left) + abs(right)):
                    break

                if not f1_ok:
                    left = x1
                    x1 = right - gr * (right - left)
                    f1_ok, f1, _ = eval_dual_and_lam(x1)
                    continue
                if not f2_ok:
                    right = x2
                    x2 = left + gr * (right - left)
                    f2_ok, f2, _ = eval_dual_and_lam(x2)
                    continue

                if f1 > f2:
                    left = x1
                    x1 = x2
                    f1 = f2
                    x2 = left + gr * (right - left)
                    f2_ok, f2, _ = eval_dual_and_lam(x2)
                else:
                    right = x2
                    x2 = x1
                    f2 = f1
                    x1 = right - gr * (right - left)
                    f1_ok, f1, _ = eval_dual_and_lam(x1)

            return 0.5 * (left + right)

        # ---------- main ----------
        omega0 = find_omega_min_feasible()
        if omega0 is None:
            return {
                "omega_star": np.nan,
                "dual_value": np.nan,
                "lambda_star": np.full(n, np.nan),
                "status": "no_feasible_omega_found",
            }

        bowl = bracket_minimizer(omega0)
        if bowl is None:
            feas0, d0, lam0_val = eval_dual_and_lam(omega0)
            return {
                "omega_star": float(omega0),
                "dual_value": float(d0),
                "lambda_star": np.asarray(lam0_val) if lam0_val is not None else np.full(n, np.nan),
                "status": "failed_to_bracket_minimizer_return_boundary",
            }

        pt_a, pt_b, pt_c, fa, fb, fc = bowl

        if pt_b == pt_a:
            omega_star = pt_a
        else:
            omega_star = golden_section(pt_a, pt_b, pt_c)

        feas, dstar, lam_star = eval_dual_and_lam(omega_star)

        return {
            "omega_star": float(omega_star),
            "dual_value": float(dstar),
            "lambda_star": np.asarray(lam_star) if lam_star is not None else np.full(n, np.nan),
            "domain_feasible": bool(feas),
            "omega_min_feasible": float(omega0),
            "bracket": (float(pt_a), float(pt_b), float(pt_c)),
            "status": "optimal_dual_1d_golden",
        }

    def optimize_dual_sdp_lambda_t(
        self,
        solver=None,
        solver_opts=None,
        tol: float = 1e-8,
    ):
        """
        Solve the single-constraint QCQP via the SDP dual in (omega, d):

            min_{omega >= 0, d in R}  d + c - omega*(c_c - r)
            s.t.  [[omega*A_c - A,        1/2*(omega*b_c - b)],
                   [1/2*(omega*b_c - b)^T,  d                ]]  ⪰ 0

        Recover lambda via KKT stationarity:
            lambda = -1/2 * M(omega*)^† * s(omega*)
        where M(omega) = omega*A_c - A,  s(omega) = omega*b_c - b.
        """
        try:
            import cvxpy as cp
        except Exception as e:
            raise ImportError(
                "cvxpy is required for optimize_dual_sdp_lambda_t(). "
                "Install it (and a solver like SCS/ECOS/MOSEK)."
            ) from e

        n = self.A.shape[0]
        if self.A.shape != (n, n) or self.A_c.shape != (n, n):
            raise ValueError("A and A_c must be square matrices of the same shape.")
        if self.b.shape != (n,) or self.b_c.shape != (n,):
            raise ValueError("b and b_c must be vectors of length natparamdim.")

        r = self.r

        omega = cp.Variable(nonneg=True)
        d = cp.Variable()

        M_expr = omega * self.A_c - self.A
        s_expr = 0.5 * (omega * self.b_c - self.b)
        LMI = cp.bmat([
            [M_expr,                             cp.reshape(s_expr, (n, 1))],
            [cp.reshape(s_expr, (1, n)), cp.reshape(d, (1, 1))]
        ])

        objective = cp.Minimize(d + self.c - omega * (self.c_c - r))
        constraints = [LMI >> 0]

        prob = cp.Problem(objective, constraints)

        solve_kwargs = {}
        if solver is not None:
            solve_kwargs["solver"] = solver
        if solver_opts:
            solve_kwargs.update(solver_opts)

        try:
            prob.solve(**solve_kwargs)
        except Exception:
            prob.solve(solver=cp.SCS)

        if prob.value is None or omega.value is None or d.value is None:
            return {}

        omega_star = float(omega.value)
        dual_value = float(prob.value)

        # Recover lambda via KKT: lambda = -1/2 * M(omega*)^† * s(omega*)
        # When s = 0 (b = b_c = 0), lambda lies in ker(M(omega*)) instead.
        M = omega_star * self.A_c - self.A
        s = omega_star * self.b_c - self.b
        M_pinv = np.linalg.pinv(self._sym(M), rcond=tol)
        lam_star = -0.5 * M_pinv @ s
        if np.linalg.norm(s) <= tol * max(float(np.linalg.norm(self.b)), 1.0):
            lam_star = self._complete_lambda_from_kernel(lam_star, M, tol)
        lam_star = self._canonical_sign(lam_star)

        primal_value = lam_star @ self.A @ lam_star + lam_star @ self.b + self.c

        return {
            "dual_value": dual_value,
            "lambda_star": lam_star,
            "primal_value": primal_value,
            "omega_star": omega_star,
        }

    def optimize_through_generalized_eigenvalue(self, nugget: float = 1e-10):
        """
        Solve the QCQP via the generalised eigenvalue problem.

        Valid when the base measure g = π_ref, which makes b = b_c = 0 and
        c = c_c = 0.  The problem then simplifies to:

            sup_{λᵀ A_c λ ≤ r} λᵀ A λ  =  r · ω_max

        where ω_max is the largest generalised eigenvalue of  A λ' = ω A_c λ'.
        The optimal solution is  λ_star = sqrt(r) · λ'_star,  with λ'_star the
        A_c-normalised eigenvector for ω_max.  Complexity O(K³).

        Args:
            nugget: regularisation added to A_c when it is not PD.
        """
        from scipy.linalg import eigh as scipy_eigh
        import warnings

        b_tol = 1e-6 * max(float(np.trace(np.abs(self.A))), 1.0)
        if float(np.linalg.norm(self.b)) > b_tol:
            warnings.warn(
                f"||b|| = {np.linalg.norm(self.b):.2e} is non-negligible (tol {b_tol:.2e}). "
                "Generalised eigenvalue method assumes b = b_c = 0 (g = π_ref)."
            )

        A = self._sym(self.A)
        A_c = self._sym(self.A_c)

        min_eig = float(np.linalg.eigvalsh(A_c).min())
        if min_eig < nugget:
            shift = nugget - min_eig
            A_c = A_c + shift * np.eye(self.d)
            print(f"A_c shifted by {shift:.2e} to ensure PD for generalised eigenvalue solver.")

        # scipy_eigh solves A v = ω A_c v and returns A_c-orthonormal eigenvectors
        # in ascending eigenvalue order; the optimum is the last column.
        omega_vals, V = scipy_eigh(A, A_c)

        omega_star = float(omega_vals[-1])
        lam_prime = V[:, -1]  # A_c-normalised: lam_prime.T @ A_c @ lam_prime = 1

        lam_star = np.sqrt(max(self.r, 0.0)) * lam_prime
        lam_star = self._canonical_sign(lam_star)

        primal_value = self._evaluate_qf(lam_star)
        constraint_value = self._evaluate_constraint_qf(lam_star)

        return {
            "lambda_star": lam_star,
            "omega_star": omega_star,
            "primal_value": primal_value,
            "constraint_value": constraint_value,
            "theoretical_value": float(self.r) * omega_star,
        }
