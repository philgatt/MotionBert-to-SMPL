"""Portable MP4 previews without a display server or OpenGL dependency."""
from __future__ import annotations

import cv2
import numpy as np

H36M_EDGES = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),
             (9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
HALPE_EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
              (11,13),(13,15),(12,14),(14,16),(0,17),(17,18),(18,19)]

# Matplotlib's default 3D camera leaves too much empty space around an upright body.
# A shorter camera distance provides a useful close-up while the small fit margin
# still keeps hands, feet and the head inside the frame.
MESH_CAMERA_DISTANCE = 4.5
MESH_FIT_MARGIN = 1.0


def writer(path, fps, size):
    result = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not result.isOpened():
        raise RuntimeError(f"Could not create preview: {path}")
    return result


def overlay(video, coords, fps, destination):
    cap = cv2.VideoCapture(str(video))
    out = None
    try:
        for points in coords:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Video ended before the keypoint timeline during rendering.")
            if out is None:
                out = writer(destination, fps, (frame.shape[1], frame.shape[0]))
            for a, b in HALPE_EDGES:
                if min(points[a,2], points[b,2]) > 0.2:
                    cv2.line(frame, tuple(points[a,:2].astype(int)), tuple(points[b,:2].astype(int)), (80,220,90), 2)
            for x, y, confidence in points:
                if confidence > 0.2:
                    cv2.circle(frame, (round(float(x)), round(float(y))), 3, (60,180,255), -1)
            out.write(frame)
    finally:
        cap.release()
        if out is not None:
            out.release()


def preview3d(data, fps, destination, faces=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    bound = 0.01
    for points in data:
        centered = points if faces is not None else points - points[:1]
        bound = max(bound, float(np.max(np.abs(centered))))
    bound *= MESH_FIT_MARGIN if faces is not None else 1.1
    out = writer(destination, fps, (600,600))
    try:
        for index, points in enumerate(data):
            ax.clear()
            p = points.copy() if faces is not None else points - points[:1]
            # The thesis pose outputs and MotionBERT predictions use camera y-down.
            p = p[:, [0,2,1]]
            p[:,2] *= -1
            if faces is None:
                for a,b in H36M_EDGES:
                    ax.plot(*p[[a,b]].T, color="#167d9a", linewidth=2)
            else:
                triangles = p[faces]
                normals = np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0])
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
                light = np.array([0.3,-0.7,0.6])
                shade = 0.4 + 0.6 * np.clip(normals @ (light / np.linalg.norm(light)), 0, 1)
                colors = shade[:,None] * np.array([0.35,0.64,0.74])
                ax.add_collection3d(Poly3DCollection(triangles, facecolor=colors, edgecolor="none", alpha=1))
            ax.set(xlim=(-bound,bound), ylim=(-bound,bound), zlim=(-bound,bound),
                   title=f"{'SMPL' if faces is not None else 'MotionBERT'} · frame {index}")
            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect((1,1,1))
            if faces is not None:
                ax.dist = MESH_CAMERA_DISTANCE
            ax.view_init(elev=12, azim=-65)
            ax.set_axis_off()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            out.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
    finally:
        out.release()
        plt.close(fig)


def render(output, video, info, selections):
    if "2d" in selections:
        overlay(video, np.load(output / "keypoints_2d.npy"), info["fps"], output / "preview_2d.mp4")
    if "3d" in selections:
        preview3d(np.load(output / "motionbert.npy"), info["fps"], output / "preview_3d.mp4")
    if "mesh" in selections:
        preview3d(np.load(output / "vertices.npy"), info["fps"],
                  output / "preview_mesh.mp4", np.load(output / "faces.npy"))
