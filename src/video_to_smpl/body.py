"""Independent pose and shape choices, followed by one consistent SMPL forward pass."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from .checkpoints import load_checkpoint


def _legacy_model_compat():
    """Allow original Chumpy-based SMPL pickles on modern Python/NumPy."""
    import importlib.util
    import inspect
    if importlib.util.find_spec("chumpy") is None:
        return
    added = []
    aliases = dict(bool=bool, int=int, float=float, complex=complex,
                   object=object, str=str, unicode=str)
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
            added.append(name)
    patched_inspect = not hasattr(inspect, "getargspec")
    if patched_inspect:
        inspect.getargspec = inspect.getfullargspec
    try:
        import chumpy  # noqa: F401 - initialize legacy deserialization classes
    finally:
        for name in added:
            delattr(np, name)
        if patched_inspect:
            delattr(inspect, "getargspec")


def load_smpl(config, device):
    from smplx import SMPL
    _legacy_model_compat()
    model = SMPL(config["smpl_model"], gender="neutral", batch_size=1,
                 create_transl=False).double().to(device).eval()
    reg = np.load(config["h36m_regressor"], allow_pickle=False)
    if reg.shape != (17, model.v_template.shape[0]) or not np.isfinite(reg).all():
        raise ValueError("H36M regressor must have shape (17, number of SMPL vertices).")
    model.register_buffer("J_regressor_h36m", torch.tensor(reg, dtype=torch.double, device=device))
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def forward(model, pose, betas, device, matrices=None):
    if matrices is not None:
        return model(
            global_orient=torch.as_tensor(matrices[:, :1], dtype=torch.double, device=device),
            body_pose=torch.as_tensor(matrices[:, 1:], dtype=torch.double, device=device),
            betas=torch.as_tensor(betas, dtype=torch.double, device=device), pose2rot=False,
        ).vertices
    return model(
        global_orient=torch.as_tensor(pose[:, :3], dtype=torch.double, device=device),
        body_pose=torch.as_tensor(pose[:, 3:], dtype=torch.double, device=device),
        betas=torch.as_tensor(betas, dtype=torch.double, device=device),
        pose2rot=True,
    ).vertices


def rest_joints(model, device):
    with torch.no_grad():
        vertices = forward(model, np.zeros((1, 72)), np.zeros((1, 10)), device)
        return (model.J_regressor_h36m @ vertices[0]).cpu().numpy()


def checkpoint_args(checkpoint):
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else vars(args)


def learned_pose(coords, config, device):
    checkpoint = load_checkpoint(config["pose_checkpoint"])
    args = checkpoint_args(checkpoint)
    if not str(checkpoint.get("model_type", "")).startswith("pose_only_smplr_"):
        raise ValueError("Expected the released pose-only SMPLR checkpoint.")
    from .methods.pose_only_smplr import PoseOnlySMPLRRegressor
    activation = "elu" if "elu" in str(checkpoint.get("model_type", "")) else "relu"
    model = PoseOnlySMPLRRegressor(activation=activation).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    x = coords.copy()
    if config.get("pose_rx180", True):
        x *= np.array([1, -1, -1], dtype=np.float32)
    # The current thesis inference script feeds raw coordinates; the network
    # derives root-relative normalized bone directions internally.
    x = x.reshape(len(x), 51).astype(np.float32)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), config["batch_size"]):
            result = model(torch.from_numpy(x[start:start + config["batch_size"]]).to(device))
            raw = result if torch.is_tensor(result) else result.pred_theta_mats_raw
            chunks.append(raw.cpu().numpy())
    matrices = np.concatenate(chunks)
    return matrices_to_axis_angle(matrices), matrices


def matrices_to_axis_angle(matrices):
    """Export a proper SO(3) approximation without changing the raw network output."""
    from scipy.spatial.transform import Rotation
    u, _, vt = np.linalg.svd(matrices.astype(np.float64))
    correction = np.broadcast_to(np.eye(3), matrices.shape).copy()
    correction[..., 2, 2] = np.linalg.det(u @ vt)
    rotations = u @ correction @ vt
    return Rotation.from_matrix(rotations.reshape(-1,3,3)).as_rotvec().reshape(-1,72)


def temporal_shape(coords, config, device):
    from .methods.shape_inference import _model_from_checkpoint, _predict_track
    checkpoint = load_checkpoint(config["shape_checkpoint"])
    args = checkpoint_args(checkpoint)
    architecture = str(args.get("architecture", checkpoint.get("architecture", "simple")))
    if architecture != "staged":
        raise ValueError(f"Expected the released staged shape checkpoint, received {architecture!r}.")
    model = _model_from_checkpoint(checkpoint).to(device).eval()
    x = coords.copy()
    if config.get("shape_root_center", False):
        x -= x[:, :1]
    if config.get("shape_rx180", False):
        x *= np.array([1, -1, -1], dtype=np.float32)
    clip_len = int(args.get("clip_len", 50))
    stride = config["shape_stride"]
    if not 1 <= stride <= clip_len:
        raise ValueError("shape_stride must be between 1 and the checkpoint clip length.")
    mode = str(args.get("clip_coordinate_mode", checkpoint.get("clip_coordinate_mode", "as_is")))
    if mode not in ("as_is", "frame0_pelvis_origin"):
        raise ValueError(f"Unsupported shape coordinate mode: {mode}")
    return _predict_track(model, x, np.arange(len(x)), clip_len, stride,
                          config["batch_size"], torch.device(device), mode)


def recover(coords, pose_method, shape_method, config, device):
    model = load_smpl(config, device)
    rest = rest_joints(model, device)
    diagnostics = {}
    matrices = None
    if pose_method == "kinematic":
        from .methods.kinematics import calculate_smpl_params_from_x3d_motionbert_bvh_rodrigues
        _, pose = calculate_smpl_params_from_x3d_motionbert_bvh_rodrigues(coords, rest)
    elif pose_method == "optimization":
        from .methods.optimization import _run_pose_to_smpl_fit_from_x3d
        chunks = []
        losses = []
        # Each frame has an independent pose; chunking bounds SMPL autograd memory.
        for start in range(0, len(coords), config["batch_size"]):
            _, p, _, diag = _run_pose_to_smpl_fit_from_x3d(
                coords[start:start + config["batch_size"]], smpl_model=model,
                device=torch.device(device), iters=config["optimization_iterations"],
                lr=float(config.get("optimization_lr", 0.01)), optimize_shape=False,
                optimize_scale=False, fit_loss="mse_h36m17", pre_scale_to_rest=False,
                smpl_rest_h36m17=rest, motionbert_rx180_flip_input=False)
            chunks.append(p)
            losses.append(diag["final_loss"])
        pose = np.concatenate(chunks)
        diagnostics["optimization_final_mse_by_batch"] = losses
    elif pose_method == "learned":
        pose, matrices = learned_pose(coords, config, device)
        diagnostics["pose_representation"] = "raw network matrices; axis-angle export is an SO(3) approximation"
    else:
        raise ValueError(f"Unknown pose method: {pose_method}")
    if shape_method == "temporal":
        betas = temporal_shape(coords, config, device)
    elif shape_method == "neutral":
        betas = np.zeros((len(coords), 10), dtype=np.float32)
    else:
        raise ValueError(f"Unknown shape method: {shape_method}")
    if pose.shape != (len(coords), 72) or betas.shape != (len(coords), 10):
        raise ValueError("Pose or shape prediction has an invalid shape.")
    if not np.isfinite(pose).all() or not np.isfinite(betas).all():
        raise ValueError("Pose or shape inference produced nonfinite values.")
    return model, pose.astype(np.float32), betas.astype(np.float32), diagnostics, matrices


def save_meshes(model, pose, betas, config, device, output, matrices=None):
    """Pelvis-centered meshes; monocular root coordinates are saved separately."""
    vertex_count = int(model.v_template.shape[0])
    vertices_path = output / "vertices.npy"
    try:
        vertices = np.lib.format.open_memmap(vertices_path, mode="w+", dtype=np.float32,
                                             shape=(len(pose), vertex_count, 3))
        memory_backed = False
    except PermissionError:
        # Some Windows sandboxed Python builds deny Path.resolve() inside
        # NumPy's memmap implementation. Keep short runs functional there.
        vertices = np.empty((len(pose), vertex_count, 3), dtype=np.float32)
        memory_backed = True
    joints = np.empty((len(pose), 17, 3), dtype=np.float32)
    translations = np.empty((len(pose), 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(pose), config["batch_size"]):
            end = min(len(pose), start + config["batch_size"])
            v = forward(model, pose[start:end], betas[start:end], device,
                        None if matrices is None else matrices[start:end])
            j = model.J_regressor_h36m @ v
            root = j[:, :1]
            vertices[start:end] = (v - root).cpu().numpy()
            joints[start:end] = (j - root).cpu().numpy()
            translations[start:end] = -root[:, 0].cpu().numpy()
    if memory_backed:
        np.save(vertices_path, vertices)
    else:
        vertices.flush()
    del vertices
    np.save(output / "smpl_joints.npy", joints)
    np.save(output / "faces.npy", np.asarray(model.faces, dtype=np.int32))
    return translations
