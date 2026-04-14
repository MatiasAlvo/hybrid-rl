import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

from src import np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_matrix_store_dir() -> Path:
    return _repo_root() / "configs" / "matrices"


def _stable_seed(n_stores: int, n_modes: int, lambda_u: float) -> int:
    seed_input = f"{n_stores}|{n_modes}|{lambda_u}".encode("utf-8")
    digest = hashlib.sha256(seed_input).digest()
    return int.from_bytes(digest[:4], "little")


def _format_param(value: float) -> str:
    return f"{value}".replace(".", "p")


def _matrix_filename(
    n_stores: int,
    n_modes: int,
    lambda_u: float,
    instability: float,
    coupling_strength: float,
    b_high: float,
    b_low: float,
) -> str:
    lambda_str = _format_param(lambda_u)
    inst_str = _format_param(instability)
    coupling_str = _format_param(coupling_strength)
    b_high_str = _format_param(b_high)
    b_low_str = _format_param(b_low)
    return (
        f"lqr_d{n_stores}_m{n_modes}_lambda{lambda_str}"
        f"_inst{inst_str}_cpl{coupling_str}_bh{b_high_str}_bl{b_low_str}.npz"
    )


def _legacy_matrix_filename(n_stores: int, n_modes: int, lambda_u: float) -> str:
    lambda_str = _format_param(lambda_u)
    return f"lqr_d{n_stores}_m{n_modes}_lambda{lambda_str}.npz"


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def generate_switched_lqr_matrices(
    n_stores: int,
    n_modes: int,
    lambda_u: float = 0.1,
    seed: Optional[int] = None,
    instability: float = 1.1,
    coupling_strength: float = 0.05,
    b_high: float = 1.0,
    b_low: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if seed is None:
        seed = _stable_seed(n_stores, n_modes, lambda_u)

    rng = np.random.default_rng(seed)

    noise = rng.standard_normal((n_stores, n_stores))
    noise = (noise + noise.T) / 2.0
    max_abs = np.max(np.abs(noise))
    if max_abs > 0:
        noise = noise / max_abs

    A_base = instability * np.eye(n_stores) + coupling_strength * noise

    indices = np.arange(n_stores)
    groups = np.array_split(indices, n_modes)

    B_list = []
    for group in groups:
        B_mode = np.eye(n_stores) * b_low
        for idx in group:
            B_mode[idx, idx] = b_high
        B_list.append(B_mode)

    A = np.repeat(A_base[None, :, :], n_modes, axis=0)
    B = np.stack(B_list, axis=0)
    Q = np.repeat(np.eye(n_stores)[None, :, :], n_modes, axis=0)
    R = np.repeat((lambda_u * np.eye(n_stores))[None, :, :], n_modes, axis=0)

    return A, B, Q, R


def load_matrices_from_npz(path_value: str) -> Dict[str, np.ndarray]:
    path = _resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"LQR matrix file not found: {path}")
    data = np.load(path, allow_pickle=False)
    return {
        "A": data["A"],
        "B": data["B"],
        "Q": data["Q"],
        "R": data["R"],
    }


def save_matrices_to_npz(
    path: Path, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, A=A, B=B, Q=Q, R=R)


def load_or_generate_matrices(
    n_stores: int,
    n_modes: int,
    lambda_u: float,
    seed: Optional[int],
    cache_dir: Optional[str],
    instability: float,
    coupling_strength: float,
    b_high: float,
    b_low: float,
) -> Dict[str, np.ndarray]:
    base_dir = _default_matrix_store_dir() if cache_dir is None else _resolve_path(cache_dir)
    file_path = base_dir / _matrix_filename(
        n_stores,
        n_modes,
        lambda_u,
        instability,
        coupling_strength,
        b_high,
        b_low,
    )
    if file_path.exists():
        return load_matrices_from_npz(str(file_path))

    legacy_path = base_dir / _legacy_matrix_filename(n_stores, n_modes, lambda_u)
    if legacy_path.exists():
        return load_matrices_from_npz(str(legacy_path))

    A, B, Q, R = generate_switched_lqr_matrices(
        n_stores=n_stores,
        n_modes=n_modes,
        lambda_u=lambda_u,
        seed=seed,
        instability=instability,
        coupling_strength=coupling_strength,
        b_high=b_high,
        b_low=b_low,
    )
    save_matrices_to_npz(file_path, A, B, Q, R)
    return {"A": A, "B": B, "Q": Q, "R": R}


def _matrix_mode_count(matrix_value) -> Optional[int]:
    if matrix_value is None:
        return None
    if hasattr(matrix_value, "shape"):
        if len(matrix_value.shape) >= 1:
            return int(matrix_value.shape[0])
        return None
    if isinstance(matrix_value, (list, tuple)):
        return len(matrix_value)
    return None


def _validate_lqr_modes(problem_params: Dict, n_modes: int, lqr_params: Dict) -> None:
    if problem_params.get("skip_lqr_mode_validation"):
        return
    discrete_features = problem_params.get("discrete_features", {})
    lqr_mode = discrete_features.get("lqr_mode", {})
    values = lqr_mode.get("values")
    if isinstance(values, list) and len(values) != n_modes:
        raise ValueError(
            f"lqr_mode.values length ({len(values)}) must match n_modes ({n_modes})."
        )

    thresholds = lqr_mode.get("thresholds")
    if isinstance(thresholds, list) and len(thresholds) not in (n_modes, n_modes + 1):
        raise ValueError(
            f"lqr_mode.thresholds length ({len(thresholds)}) must match n_modes ({n_modes}) "
            "or n_modes + 1."
        )

    a_modes = _matrix_mode_count(lqr_params.get("A"))
    if a_modes is not None and a_modes != n_modes:
        raise ValueError(
            f"LQR A mode count ({a_modes}) must match n_modes ({n_modes})."
        )


def prepare_lqr_matrices(problem_params: Dict) -> None:
    if problem_params.get("simulator_type") != "lqr_hybrid":
        return

    lqr_params = problem_params.get("lqr", {})
    if all(key in lqr_params for key in ("A", "B", "Q", "R")):
        n_modes = problem_params.get("n_modes")
        if n_modes is not None:
            _validate_lqr_modes(problem_params, n_modes, lqr_params)
        return

    n_stores = problem_params.get("n_stores")
    if n_stores is None:
        raise ValueError("LQR matrix generation requires problem_params.n_stores.")

    n_modes = problem_params.get("n_modes")
    if n_modes is None:
        raise ValueError("LQR matrix generation requires problem_params.n_modes.")

    if "source_path" in lqr_params and lqr_params["source_path"]:
        matrices = load_matrices_from_npz(lqr_params["source_path"])
    else:
        lambda_u = lqr_params.get("lambda_u", 0.1)
        seed = lqr_params.get("seed")
        cache_dir = lqr_params.get("cache_dir")
        instability = lqr_params.get("instability", 1.1)
        coupling_strength = lqr_params.get("coupling_strength", 0.05)
        b_high = lqr_params.get("b_high", 1.0)
        b_low = lqr_params.get("b_low", 0.2)
        matrices = load_or_generate_matrices(
            n_stores,
            n_modes,
            lambda_u,
            seed,
            cache_dir,
            instability,
            coupling_strength,
            b_high,
            b_low,
        )

    lqr_params = {**lqr_params, **matrices}
    problem_params["lqr"] = lqr_params
    _validate_lqr_modes(problem_params, n_modes, lqr_params)
