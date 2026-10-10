"""Kinematic solver extracted from the thesis pipeline; original conventions preserved."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

_MAPPING = [
    (0, None), (4, None), (1, None), (7, None), (5, None), (2, None), (3, None), (6, None), (12, None),
    (8, None), (15, None), (None, [0, 0, 0]), (9, None), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]),
    (10, None), (13, None), (11, None), (14, None), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]),
    (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]),
    (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]),
    (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]),
    (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]),
    (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]),
    (None, [0, 0, 0]), (None, [0, 0, 0]), (None, [0, 0, 0]),
]

_BODY_PARTS = [
    "Hip", "RightHip", "RightKnee", "RightAnkle", "LeftHip", "LeftKnee", "LeftAnkle",
    "Spine", "Thorax", "Neck", "LeftShoulder", "LeftElbow", "LeftWrist",
    "RightShoulder", "RightElbow", "RightWrist",
]

def _require_scipy(feature: str) -> None:
    if least_squares is None or R is None:
        raise ImportError(
            f"SciPy is required for {feature}, but could not be imported in this environment."
        )

def _euler_to_quaternion_zxy(euler_z: float, euler_x: float, euler_y: float) -> Tuple[float, float, float, float]:
    _require_scipy("direct_kinematics quaternion conversion")
    rot = R.from_euler("zxy", [euler_z, euler_x, euler_y], degrees=True)
    quat = rot.as_quat()
    return float(quat[3]), float(quat[1]), float(quat[2]), float(quat[0])

def _rodrigues_from_quat_wxyz(quat_wxyz: Tuple[float, float, float, float]) -> np.ndarray:
    _require_scipy("direct_kinematics Rodrigues conversion")
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return rot.as_rotvec()

def _apply_mapping(flat_values: List[float]) -> List[float]:
    grouped = [flat_values[i : i + 3] for i in range(0, len(flat_values), 3)]
    out_groups: List[List[float]] = []
    for index, padding in _MAPPING:
        if index is not None:
            out_groups.append(grouped[index])
        elif padding is not None:
            out_groups.append(padding)
    return [item for group in out_groups for item in group]

def _channels_to_smpl_translations_and_thetas(
    channels: List[List[float]],
    local_frame_correction_x_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    all_translations: List[List[float]] = []
    all_thetas: List[List[float]] = []
    corr_mat: Optional[np.ndarray] = None
    if abs(float(local_frame_correction_x_deg)) > 1e-12:
        _require_scipy("channel->SMPL local-frame correction")
        corr_mat = R.from_euler("x", float(local_frame_correction_x_deg), degrees=True).as_matrix()
    for channel in channels:
        all_translations.append(channel[0:3])
        rotation_values = channel[3:]
        pose = [0.0] * (len(_BODY_PARTS) * 3)
        for i in range(len(_BODY_PARTS)):
            if corr_mat is None:
                # Preserve original direct_kinematics channel->theta convention.
                quat = _euler_to_quaternion_zxy(
                    -rotation_values[i * 3 + 2],
                    rotation_values[i * 3 + 1],
                    rotation_values[i * 3],
                )
                rod = _rodrigues_from_quat_wxyz(quat)
            else:
                _require_scipy("channel->SMPL rotation conversion")
                rot_local = R.from_euler(
                    "zxy",
                    [
                        rotation_values[i * 3],
                        rotation_values[i * 3 + 1],
                        -rotation_values[i * 3 + 2],
                    ],
                    degrees=True,
                ).as_matrix()
                # Convert from a backend-local rest frame to the direct-kinematics
                # frame expected by the shared channel->SMPL mapping.
                rot_local = corr_mat @ rot_local @ corr_mat.T
                rod = R.from_matrix(rot_local).as_rotvec()
            pose[i * 3 + 0] = float(rod[2])
            pose[i * 3 + 1] = float(rod[0])
            pose[i * 3 + 2] = float(rod[1])
        reordered_theta = _apply_mapping(pose)
        all_thetas.append(reordered_theta[0:72])
    return np.asarray(all_translations, dtype=np.float64), np.asarray(all_thetas, dtype=np.float64)

_M2B_EPS = 1e-8

_M2B_ORDER = [
    "Hip",
    "RightHip", "RightKnee", "RightAnkle",
    "LeftHip", "LeftKnee", "LeftAnkle",
    "Spine", "Thorax", "Neck",
    "LeftShoulder", "LeftElbow", "LeftWrist",
    "RightShoulder", "RightElbow", "RightWrist",
]

_M2B_PARENT = {
    "Hip": None,
    "RightHip": "Hip",
    "RightKnee": "RightHip",
    "RightAnkle": "RightKnee",
    "LeftHip": "Hip",
    "LeftKnee": "LeftHip",
    "LeftAnkle": "LeftKnee",
    "Spine": "Hip",
    "Thorax": "Spine",
    "Neck": "Thorax",
    "LeftShoulder": "Thorax",
    "LeftElbow": "LeftShoulder",
    "LeftWrist": "LeftElbow",
    "RightShoulder": "Thorax",
    "RightElbow": "RightShoulder",
    "RightWrist": "RightElbow",
}

_M2B_CHILDREN = {
    "Hip": ["RightHip", "LeftHip", "Spine"],
    "RightHip": ["RightKnee"],
    "RightKnee": ["RightAnkle"],
    "RightAnkle": [],
    "LeftHip": ["LeftKnee"],
    "LeftKnee": ["LeftAnkle"],
    "LeftAnkle": [],
    "Spine": ["Thorax"],
    "Thorax": ["Neck", "LeftShoulder", "RightShoulder"],
    "Neck": [],
    "LeftShoulder": ["LeftElbow"],
    "LeftElbow": ["LeftWrist"],
    "LeftWrist": [],
    "RightShoulder": ["RightElbow"],
    "RightElbow": ["RightWrist"],
    "RightWrist": [],
}

_M2B_IDX = {
    "Hip": 0,
    "RightHip": 1,
    "RightKnee": 2,
    "RightAnkle": 3,
    "LeftHip": 4,
    "LeftKnee": 5,
    "LeftAnkle": 6,
    "Spine": 7,
    "Thorax": 8,
    "Neck": 9,
    "Head": 10,
    "LeftShoulder": 11,
    "LeftElbow": 12,
    "LeftWrist": 13,
    "RightShoulder": 14,
    "RightElbow": 15,
    "RightWrist": 16,
}

def _m2b_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < _M2B_EPS:
        return np.asarray(v, dtype=np.float64).copy()
    return np.asarray(v, dtype=np.float64) / n

def _m2b_rodrigues_from_a_to_b(a: np.ndarray, b: np.ndarray, solver: str = "angle") -> np.ndarray:
    a_n = _m2b_normalize(a)
    b_n = _m2b_normalize(b)
    if np.linalg.norm(a_n) < _M2B_EPS or np.linalg.norm(b_n) < _M2B_EPS:
        return np.eye(3, dtype=np.float64)
    cross = np.cross(a_n, b_n)
    s = np.linalg.norm(cross)
    c = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    if s < _M2B_EPS:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a_n, basis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _m2b_normalize(np.cross(a_n, basis))
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + np.sin(np.pi) * K + (1.0 - np.cos(np.pi)) * (K @ K)
    axis = cross / s
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    if solver == "sincos":
        sin_t = float(s)
        cos_t = float(c)
        return np.eye(3, dtype=np.float64) + sin_t * K + (1.0 - cos_t) * (K @ K)
    theta = float(np.arccos(c))
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

def _m2b_rotation_matrix_to_euler_zxy_degrees(r_mat: np.ndarray) -> np.ndarray:
    sx = float(np.clip(r_mat[2, 1], -1.0, 1.0))
    x = float(np.arcsin(sx))
    cx = float(np.cos(x))
    if abs(cx) > 1e-6:
        z = float(np.arctan2(-r_mat[0, 1], r_mat[1, 1]))
        y = float(np.arctan2(-r_mat[2, 0], r_mat[2, 2]))
    else:
        z = float(np.arctan2(r_mat[1, 0], r_mat[0, 0]))
        y = 0.0
    return np.degrees(np.array([z, x, y], dtype=np.float64))

def _m2b_build_pelvis_frame(frame: np.ndarray) -> np.ndarray:
    hip = frame[_M2B_IDX["Hip"]]
    rhip = frame[_M2B_IDX["RightHip"]]
    lhip = frame[_M2B_IDX["LeftHip"]]
    thorax = frame[_M2B_IDX["Thorax"]]
    x_axis = _m2b_normalize(lhip - rhip)
    y_guess = _m2b_normalize(thorax - hip)
    z_axis = _m2b_normalize(np.cross(x_axis, y_guess))
    y_axis = _m2b_normalize(np.cross(z_axis, x_axis))
    if (
        np.linalg.norm(x_axis) < _M2B_EPS
        or np.linalg.norm(y_axis) < _M2B_EPS
        or np.linalg.norm(z_axis) < _M2B_EPS
    ):
        return np.eye(3, dtype=np.float64)
    return np.stack([x_axis, y_axis, z_axis], axis=1)

def calculate_smpl_params_from_x3d_motionbert_bvh_rodrigues(
    data: np.ndarray,
    smpl_rest_joints_h36m17: np.ndarray,
    root_solver: str = "pelvis_frame",
    head_twist_angle: str = "atan2",
    head_twist_formulation: str = "angle",
    swing_solver: str = "angle",
    return_channels: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[List[float]]]]:
    """
    MotionBERT->BVH path aligned to MotionBERT/tools/motionbert_to_bvh.py:
    - per-joint local rotations from rest->pose Rodrigues solve
    - BVH ZXY Euler channels in the same hierarchy/order
    - reuse existing channel->SMPL theta conversion.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 17 or arr.shape[2] < 3:
        raise ValueError(f"Unexpected H36M data shape for motionbert_bvh_rodrigues: {arr.shape}")
    motion = arr[:, :17, :3]
    n_frames = motion.shape[0]
    rest_joints = np.asarray(smpl_rest_joints_h36m17, dtype=np.float64)
    if rest_joints.shape != (17, 3):
        raise ValueError(f"Expected smpl_rest_joints_h36m17 shape (17,3), got {rest_joints.shape}")
    # Experimental variant: do not frame-1 similarity-align the sequence to rest.
    motion_norm = np.asarray(motion, dtype=np.float64).copy()
    root_origin = motion_norm[0, _M2B_IDX["Hip"]].copy()
    primary_child = {
        "Hip": "Spine",
        "RightHip": "RightKnee",
        "RightKnee": "RightAnkle",
        "LeftHip": "LeftKnee",
        "LeftKnee": "LeftAnkle",
        "Spine": "Thorax",
        # Structural thorax bone should point to head-end.
        "Thorax": "Head",
        "LeftShoulder": "LeftElbow",
        "LeftElbow": "LeftWrist",
        "RightShoulder": "RightElbow",
        "RightElbow": "RightWrist",
    }
    rest_dir = {j: rest_joints[_M2B_IDX[c]] - rest_joints[_M2B_IDX[j]] for j, c in primary_child.items()}
    rest_root_frame = _m2b_build_pelvis_frame(rest_joints)
    root_children = ("Spine", "LeftHip", "RightHip")
    channels: List[List[float]] = []
    prev_neck_twist: Optional[float] = None
    for t in range(n_frames):
        frame = motion_norm[t]
        local_r: Dict[str, np.ndarray] = {}
        global_r: Dict[str, np.ndarray] = {}
        if root_solver == "hybrik_svd":
            rest_cols = [_m2b_normalize(rest_joints[_M2B_IDX[c]] - rest_joints[_M2B_IDX["Hip"]]) for c in root_children]
            curr_cols = [_m2b_normalize(frame[_M2B_IDX[c]] - frame[_M2B_IDX["Hip"]]) for c in root_children]
            r_root = _rigid_align_svd(rest_cols, curr_cols)
        else:
            curr_root_frame = _m2b_build_pelvis_frame(frame)
            r_root = curr_root_frame @ rest_root_frame.T
        local_r["Hip"] = r_root
        global_r["Hip"] = r_root
        for name in _M2B_ORDER[1:]:
            parent = _M2B_PARENT[name]
            assert parent is not None
            if name == "Neck":
                # Nose-driven look-at: solve full neck orientation so face forward
                # follows the Head->Nose cue, not only an axial twist.
                r_parent_global = global_r[parent]

                rest_up_global = rest_joints[_M2B_IDX["Head"]] - rest_joints[_M2B_IDX["Thorax"]]
                rest_fwd_global = rest_joints[_M2B_IDX["Neck"]] - rest_joints[_M2B_IDX["Head"]]
                curr_up_global = frame[_M2B_IDX["Head"]] - frame[_M2B_IDX["Thorax"]]
                curr_fwd_global = frame[_M2B_IDX["Neck"]] - frame[_M2B_IDX["Head"]]

                rest_up = _m2b_normalize(rest_up_global)
                curr_up = _m2b_normalize(curr_up_global)
                rest_fwd = _m2b_normalize(rest_fwd_global)
                curr_fwd = _m2b_normalize(curr_fwd_global)

                # Orthogonalize forward against up.
                rest_fwd = rest_fwd - float(np.dot(rest_fwd, rest_up)) * rest_up
                curr_fwd = curr_fwd - float(np.dot(curr_fwd, curr_up)) * curr_up
                rest_fwd = _m2b_normalize(rest_fwd)
                curr_fwd = _m2b_normalize(curr_fwd)

                if (
                    np.linalg.norm(rest_up) < _M2B_EPS
                    or np.linalg.norm(curr_up) < _M2B_EPS
                    or np.linalg.norm(rest_fwd) < _M2B_EPS
                    or np.linalg.norm(curr_fwd) < _M2B_EPS
                ):
                    local_r[name] = np.eye(3, dtype=np.float64)
                    global_r[name] = r_parent_global @ local_r[name]
                    continue

                rest_right = _m2b_normalize(np.cross(rest_up, rest_fwd))
                curr_right = _m2b_normalize(np.cross(curr_up, curr_fwd))
                if np.linalg.norm(rest_right) < _M2B_EPS or np.linalg.norm(curr_right) < _M2B_EPS:
                    local_r[name] = np.eye(3, dtype=np.float64)
                    global_r[name] = r_parent_global @ local_r[name]
                    continue

                # Re-orthogonalize forward for numerical stability.
                rest_fwd = _m2b_normalize(np.cross(rest_right, rest_up))
                curr_fwd = _m2b_normalize(np.cross(curr_right, curr_up))

                b_rest = np.stack([rest_right, rest_up, rest_fwd], axis=1)
                b_curr = np.stack([curr_right, curr_up, curr_fwd], axis=1)
                r_neck_global = b_curr @ b_rest.T
                local_r[name] = r_parent_global.T @ r_neck_global
                global_r[name] = r_parent_global @ local_r[name]
                prev_neck_twist = None
                continue
            if len(_M2B_CHILDREN[name]) == 0 or name not in primary_child:
                local_r[name] = np.eye(3, dtype=np.float64)
                global_r[name] = global_r[parent] @ local_r[name]
                continue
            r_parent_global = global_r[parent]
            rest_v_local = rest_dir[name]
            curr_v_global = frame[_M2B_IDX[primary_child[name]]] - frame[_M2B_IDX[name]]
            curr_v_local = r_parent_global.T @ curr_v_global
            r_local = _m2b_rodrigues_from_a_to_b(rest_v_local, curr_v_local, solver=swing_solver)
            local_r[name] = r_local
            global_r[name] = r_parent_global @ r_local
        row: List[float] = []
        root_pos = frame[_M2B_IDX["Hip"]] - root_origin
        row.extend([float(root_pos[0]), float(root_pos[1]), float(root_pos[2])])
        e_hip = _m2b_rotation_matrix_to_euler_zxy_degrees(local_r["Hip"])
        row.extend([float(e_hip[0]), float(e_hip[1]), float(e_hip[2])])
        for name in _M2B_ORDER[1:]:
            e = _m2b_rotation_matrix_to_euler_zxy_degrees(local_r[name])
            row.extend([float(e[0]), float(e[1]), float(e[2])])
        channels.append(row)
    # motionbert_bvh_rodrigues uses a BVH rest orientation that is rotated
    # by -90deg around X versus the direct_kinematics path. Compensate here
    # so recovered SMPL theta is in the same convention as other backends.
    pred_trans, thetas = _channels_to_smpl_translations_and_thetas(
        channels,
        local_frame_correction_x_deg=90.0,
    )
    # Backend-specific SMPL rotvec axis convention fix:
    # current -> expected per joint triplet:
    # [x, y, z] -> [y, -x, -z]
    if thetas.ndim == 2 and thetas.shape[1] >= 72:
        t = thetas[:, :72].reshape(-1, 24, 3)
        x = t[..., 0].copy()
        y = t[..., 1].copy()
        z = t[..., 2].copy()
        t[..., 0] = y
        t[..., 1] = -x
        t[..., 2] = -z
        thetas[:, :72] = t.reshape(-1, 72)
    if return_channels:
        return pred_trans, thetas, channels
    return pred_trans, thetas

def _rigid_align_svd(rest_cols: List[np.ndarray], target_cols: List[np.ndarray]) -> np.ndarray:
    A = np.concatenate([v.reshape(3, 1) for v in rest_cols], axis=1)
    B = np.concatenate([v.reshape(3, 1) for v in target_cols], axis=1)
    S = A @ B.T
    U, _, Vt = np.linalg.svd(S)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0.0:
        Vt[-1, :] *= -1.0
        Rm = Vt.T @ U.T
    return Rm
