from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Run lightweight validation and cleanup without printability analysis."""
    cleaned = mesh.copy()
    cleaned.remove_unreferenced_vertices()
    cleaned.remove_duplicate_faces()
    cleaned.remove_degenerate_faces()
    cleaned.remove_infinite_values()

    if cleaned.vertices.size:
        cleaned.vertices = np.asarray(cleaned.vertices, dtype=np.float32)
        cleaned.faces = np.asarray(cleaned.faces, dtype=np.int64)
        center = cleaned.bounding_box.centroid
        cleaned.apply_translation(-center)
        scale = float(cleaned.scale)
        if scale > 0:
            cleaned.apply_scale(1.0 / scale)

    return cleaned


def strip_visuals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return an untextured mesh so OBJ preview/import does not render as black."""
    untextured = mesh.copy()
    untextured.visual = trimesh.visual.ColorVisuals(mesh=untextured)
    untextured.visual.vertex_colors = np.tile(np.array([210, 210, 210, 255], dtype=np.uint8), (len(untextured.vertices), 1))
    return untextured


def prepare_preview_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Orient preview mesh for Gradio when TripoSR utilities are available."""
    preview = strip_visuals(mesh)
    try:
        from tsr.utils import to_gradio_3d_orientation
    except ImportError:
        return preview
    return to_gradio_3d_orientation(preview)


def load_mesh_summary(
    mesh: trimesh.Trimesh,
    obj_path: Path,
    stl_path: Optional[Path],
    processed_path: Path,
) -> str:
    bounds = mesh.bounds.tolist() if mesh.vertices.size else []
    lines = [
        "Done.",
        f"Processed image: {processed_path}",
        f"OBJ: {obj_path}",
        f"Vertices: {len(mesh.vertices):,}",
        f"Faces: {len(mesh.faces):,}",
        f"Watertight: {mesh.is_watertight}",
        f"Bounds: {bounds}",
    ]
    if stl_path:
        lines.append(f"STL: {stl_path}")
    return "\n".join(lines)
