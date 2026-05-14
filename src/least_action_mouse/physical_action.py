from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import config_value, load_model_config
from .preprocess import minimum_jerk_path


TARGET = np.array([1.0, 1.0])
COMPETITOR = np.array([-1.0, 1.0])
START = np.array([0.0, 0.0])
_CONFIG = load_model_config()


@dataclass(frozen=True)
class PhysicalActionParams:
    alpha: float = float(config_value(_CONFIG, "model.physical_action_ablation.alpha", 0.08))
    beta: float = float(config_value(_CONFIG, "model.physical_action_ablation.beta", 0.004))
    potential_scale: float = float(config_value(_CONFIG, "model.physical_action_ablation.potential_scale", 2.0))
    sigma: float = float(config_value(_CONFIG, "model.physical_action_ablation.sigma", 0.9))
    target_power: float = float(config_value(_CONFIG, "model.physical_action_ablation.target_power", 4.0))
    competitor_decay: float = float(config_value(_CONFIG, "model.physical_action_ablation.competitor_decay", 1.2))
    x_bound: float = float(config_value(_CONFIG, "model.physical_action_ablation.x_bound", 1.45))
    y_lower: float = float(config_value(_CONFIG, "model.physical_action_ablation.y_lower", -0.15))
    y_upper: float = float(config_value(_CONFIG, "model.physical_action_ablation.y_upper", 1.25))
    maxiter: int = int(config_value(_CONFIG, "model.optimizer.maxiter", 500))
    ftol: float = float(config_value(_CONFIG, "model.optimizer.ftol", 1e-9))
    maxls: int = int(config_value(_CONFIG, "model.optimizer.maxls", 50))


@dataclass(frozen=True)
class PhysicalActionGrid:
    rhos: np.ndarray
    target_paths: np.ndarray
    target_actions: np.ndarray
    converged: np.ndarray
    motor_only_path: np.ndarray
    target_only_path: np.ndarray


def precompute_physical_action_grid(
    rhos: np.ndarray,
    n_time: int,
    params: PhysicalActionParams | None = None,
) -> PhysicalActionGrid:
    """Precompute paths for the velocity/acceleration action ablation."""

    params = params or PhysicalActionParams()
    motor_only_path, _, motor_ok = solve_physical_action_path(
        0.0,
        n_time,
        params=params,
        target_strength=0.0,
    )
    target_only_path, _, target_only_ok = solve_physical_action_path(
        0.0,
        n_time,
        params=params,
        target_strength=1.0,
        initial_path=motor_only_path,
    )

    paths = []
    actions = []
    converged = []
    previous = target_only_path
    for rho in rhos:
        path, action, ok = solve_physical_action_path(
            float(rho),
            n_time,
            params=params,
            target_strength=1.0,
            initial_path=previous,
        )
        paths.append(path)
        actions.append(action)
        converged.append(ok)
        previous = path

    return PhysicalActionGrid(
        rhos=np.asarray(rhos, dtype=float),
        target_paths=np.stack(paths),
        target_actions=np.asarray(actions),
        converged=np.array([motor_ok, target_only_ok, *converged], dtype=bool),
        motor_only_path=motor_only_path,
        target_only_path=target_only_path,
    )


def solve_physical_action_path(
    rho: float,
    n_time: int,
    params: PhysicalActionParams | None = None,
    target_strength: float = 1.0,
    initial_path: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Minimize alpha|qdot|^2 + beta|qddot|^2 + target/competitor potentials."""

    params = params or PhysicalActionParams()
    if initial_path is None:
        initial_path = minimum_jerk_path(n_time)

    z0 = initial_path[1:-1].reshape(-1)
    bounds = []
    for _ in range(n_time - 2):
        bounds.extend([(-params.x_bound, params.x_bound), (params.y_lower, params.y_upper)])

    def objective(z: np.ndarray) -> float:
        path = _unpack_path(z, n_time)
        return _physical_action(path, rho, target_strength, params)

    def gradient(z: np.ndarray) -> np.ndarray:
        path = _unpack_path(z, n_time)
        return _physical_action_gradient(path, rho, target_strength, params)[1:-1].reshape(-1)

    result = minimize(
        objective,
        z0,
        jac=gradient,
        method=str(config_value(_CONFIG, "model.optimizer.method", "L-BFGS-B")),
        bounds=bounds,
        options={"maxiter": params.maxiter, "ftol": params.ftol, "maxls": params.maxls},
    )
    path = _unpack_path(result.x, n_time)
    return path, float(result.fun), bool(result.success)


def _unpack_path(z: np.ndarray, n_time: int) -> np.ndarray:
    internal = z.reshape(n_time - 2, 2)
    return np.vstack((START, internal, TARGET))


def _physical_action(
    path: np.ndarray,
    rho: float,
    target_strength: float,
    params: PhysicalActionParams,
) -> float:
    n_time = path.shape[0]
    t = np.linspace(0.0, 1.0, n_time)
    dt = 1.0 / (n_time - 1)
    velocity = np.diff(path, axis=0) / dt
    acceleration = np.diff(path, n=2, axis=0) / (dt * dt)

    target_profile = t**params.target_power
    competitor_profile = (1.0 - t) ** params.competitor_decay
    target_distance = np.sum((path - TARGET) ** 2, axis=1)
    competitor_distance = np.sum((path - COMPETITOR) ** 2, axis=1)
    potential = (
        -params.potential_scale
        * target_strength
        * target_profile
        * np.exp(-target_distance / (2.0 * params.sigma**2))
        -params.potential_scale
        * rho
        * competitor_profile
        * np.exp(-competitor_distance / (2.0 * params.sigma**2))
    )

    return float(
        0.5 * params.alpha * np.sum(velocity * velocity) * dt
        + 0.5 * params.beta * np.sum(acceleration * acceleration) * dt
        + np.sum(potential) * dt
    )


def _physical_action_gradient(
    path: np.ndarray,
    rho: float,
    target_strength: float,
    params: PhysicalActionParams,
) -> np.ndarray:
    n_time = path.shape[0]
    t = np.linspace(0.0, 1.0, n_time)
    dt = 1.0 / (n_time - 1)
    grad = np.zeros_like(path)

    displacement = np.diff(path, axis=0)
    velocity_scale = params.alpha / dt
    grad[:-1] -= velocity_scale * displacement
    grad[1:] += velocity_scale * displacement

    acceleration = np.diff(path, n=2, axis=0) / (dt * dt)
    accel_scale = params.beta / dt
    grad[:-2] += accel_scale * acceleration
    grad[1:-1] += -2.0 * accel_scale * acceleration
    grad[2:] += accel_scale * acceleration

    target_profile = t**params.target_power
    target_delta = path - TARGET
    target_distance = np.sum(target_delta * target_delta, axis=1)
    target_attraction = np.exp(-target_distance / (2.0 * params.sigma**2))
    grad += (
        params.potential_scale
        * target_strength
        * target_profile[:, None]
        * target_attraction[:, None]
        * target_delta
        / (params.sigma**2)
        * dt
    )

    competitor_profile = (1.0 - t) ** params.competitor_decay
    competitor_delta = path - COMPETITOR
    competitor_distance = np.sum(competitor_delta * competitor_delta, axis=1)
    competitor_attraction = np.exp(-competitor_distance / (2.0 * params.sigma**2))
    grad += (
        params.potential_scale
        * rho
        * competitor_profile[:, None]
        * competitor_attraction[:, None]
        * competitor_delta
        / (params.sigma**2)
        * dt
    )
    return grad
