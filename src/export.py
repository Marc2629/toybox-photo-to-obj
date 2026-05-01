from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import trimesh


def export_mesh(mesh: trimesh.Trimesh, output_dir: Path, export_stl: bool = True) -> Tuple[Path, Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obj_path = output_dir / f"toybox_mesh_{stamp}.obj"
    mesh.export(obj_path)

    stl_path = None
    if export_stl:
        stl_path = output_dir / f"toybox_mesh_{stamp}.stl"
        mesh.export(stl_path)

    return obj_path, stl_path


def export_preview_mesh(mesh: trimesh.Trimesh, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    preview_path = temp_dir / "preview.obj"
    mesh.export(preview_path)
    return preview_path
