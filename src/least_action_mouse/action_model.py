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
class ActionParams:
    alpha: float = float(config_value(_CONFIG, "model.primary_action.alpha", 1.0))
    beta: float = float(config_value(_CONFIG, "model.primary_action.beta", 0.003))
    potential_scale: float = float(config_value(_CONFIG, "model.primary_action.potential_scale", 2.0))
    sigma: float = float(config_value(_CONFIG, "model.primary_action.sigma", 0.9))
    competitor_decay: float = float(config_value(_CONFIG, "model.primary_action.competitor_decay", 1.2))
    x_bound: float = float(config_value(_CONFIG, "model.primary_action.x_bound", 1.45))
    y_lower: float = float(config_value(_CONFIG, "model.primary_action.y_lower", -0.15))
    y_upper: float = float(config_value(_CONFIG, "model.primary_action.y_upper", 1.25))
    maxiter: int = int(config_value(_CONFIG, "model.optimizer.maxiter", 500))
    ftol: float = float(config_value(_CONFIG, "model.optimizer.ftol", 1e-9))
    maxls: int = int(config_value(_CONFIG, "model.optimizer.maxls", 50))


@dataclass(frozen=True)
class ActionGrid:
    rhos: np.ndarray
    target_paths: np.ndarray
    target_actions: np.ndarray
    competitor_paths: np.ndarray
    competitor_actions: np.ndarray
    converged: np.ndarray


def precompute_action_grid(
    rhos: np.ndarray,
    n_time: int,
    params: ActionParams | None = None,
) -> ActionGrid:
    """Precompute least-action paths for a grid of competitor strengths."""

    params = params or ActionParams()
    target_paths, target_actions, target_ok = _solve_grid(rhos, n_time, TARGET, params)
    competitor_paths, competitor_actions, competitor_ok = _solve_grid(rhos, n_time, COMPETITOR, params)
    return ActionGrid(
        rhos=np.asarray(rhos, dtype=float),
        target_paths=target_paths,
        target_actions=target_actions,
        competitor_paths=competitor_paths,
        competitor_actions=competitor_actions,
        converged=np.logical_and(target_ok, competitor_ok),
    )


def action_value(path: np.ndarray, rho: float, params: ActionParams | None = None) -> float:
    params = params or ActionParams()
    motor_template = minimum_jerk_path(path.shape[0], tuple(path[-1]))
    return float(_discrete_action(path, rho, params, motor_template))


def _solve_grid(
    rhos: np.ndarray,
    n_time: int,
    endpoint: np.ndarray,
    params: ActionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = []
    actions = []
    converged = []
    previous: np.ndarray | None = None
    for rho in rhos:
        path, action, ok = solve_action_path(float(rho), n_time, endpoint, params, previous)
        paths.append(path)
        actions.append(action)
        converged.append(ok)
        previous = path
    return np.stack(paths), np.array(actions), np.array(converged, dtype=bool)


def solve_action_path(
    rho: float,
    n_time: int,
    endpoint: np.ndarray = TARGET,
    params: ActionParams | None = None,
    initial_path: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Find a stationary/minimal discrete-action path for a fixed rho."""

    params = params or ActionParams()
    endpoint = np.asarray(endpoint, dtype=float)
    motor_template = minimum_jerk_path(n_time, tuple(endpoint))
    if initial_path is None:
        initial_path = motor_template
    z0 = initial_path[1:-1].reshape(-1)

    bounds = []
    for _ in range(n_time - 2):
        bounds.extend([(-params.x_bound, params.x_bound), (params.y_lower, params.y_upper)])

    def objective(z: np.ndarray) -> float:
        path = _unpack_path(z, n_time, endpoint)
        return _discrete_action(path, rho, params, motor_template)

    def gradient(z: np.ndarray) -> np.ndarray:
        path = _unpack_path(z, n_time, endpoint)
        return _discrete_action_gradient(path, rho, params, motor_template)[1:-1].reshape(-1)

    result = minimize(
        objective,
        z0,
        jac=gradient,
        method=str(config_value(_CONFIG, "model.optimizer.method", "L-BFGS-B")),
        bounds=bounds,
        options={"maxiter": params.maxiter, "ftol": params.ftol, "maxls": params.maxls},
    )
    path = _unpack_path(result.x, n_time, endpoint)
    return path, float(result.fun), bool(result.success)


def _unpack_path(z: np.ndarray, n_time: int, endpoint: np.ndarray) -> np.ndarray:
    internal = z.reshape(n_time - 2, 2)
    return np.vstack((START, internal, endpoint))


def _discrete_action(
    path: np.ndarray,
    rho: float,
    params: ActionParams,
    motor_template: np.ndarray,
) -> float:
    n_time = path.shape[0]
    t = np.linspace(0.0, 1.0, n_time)
    dt = 1.0 / (n_time - 1)

    deformation = path - motor_template
    deformation_acceleration = np.diff(deformation, n=2, axis=0) / (dt * dt)

    competitor_profile = (1.0 - t) ** params.competitor_decay
    competitor_distance = np.sum((path - COMPETITOR) ** 2, axis=1)

    potential = (
        -params.potential_scale
        * rho
        * competitor_profile
        * np.exp(-competitor_distance / (2.0 * params.sigma**2))
    )

    return float(
        0.5 * params.alpha * np.sum(deformation * deformation) * dt
        + 0.5 * params.beta * np.sum(deformation_acceleration * deformation_acceleration) * dt
        + np.sum(potential) * dt
    )


def _discrete_action_gradient(
    path: np.ndarray,
    rho: float,
    params: ActionParams,
    motor_template: np.ndarray,
) -> np.ndarray:
    n_time = path.shape[0]
    t = np.linspace(0.0, 1.0, n_time)
    dt = 1.0 / (n_time - 1)

    deformation = path - motor_template
    deformation_acceleration = np.diff(deformation, n=2, axis=0) / (dt * dt)

    grad = params.alpha * deformation * dt
    accel_scale = params.beta / dt
    grad[:-2] += accel_scale * deformation_acceleration
    grad[1:-1] += -2.0 * accel_scale * deformation_acceleration
    grad[2:] += accel_scale * deformation_acceleration

    competitor_profile = (1.0 - t) ** params.competitor_decay
    competitor_delta = path - COMPETITOR
    competitor_distance = np.sum(competitor_delta * competitor_delta, axis=1)
    attraction = np.exp(-competitor_distance / (2.0 * params.sigma**2))
    grad += (
        params.potential_scale
        * rho
        * competitor_profile[:, None]
        * attraction[:, None]
        * competitor_delta
        / (params.sigma**2)
        * dt
    )
    return grad
