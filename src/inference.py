from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import trimesh
from PIL import Image

LOGGER = logging.getLogger(__name__)


class ImageToMeshBackend(Protocol):
    def generate(
        self,
        image: Image.Image,
        mesh_resolution: int,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
    ) -> trimesh.Trimesh:
        """Generate a mesh from a preprocessed PIL image."""


@dataclass
class TripoSRBackend:
    model_name_or_path: str = "stabilityai/TripoSR"
    device: str = "auto"
    chunk_size: int = 8192

    def __post_init__(self) -> None:
        self.device = self._resolve_device(self.device)
        try:
            from tsr.system import TSR
        except ImportError as exc:
            raise RuntimeError(
                "TripoSR is not importable. Clone the TripoSR repo and add it to PYTHONPATH, "
                "as shown in README.md."
            ) from exc

        LOGGER.info("Loading TripoSR model '%s' on %s", self.model_name_or_path, self.device)
        self.model = TSR.from_pretrained(
            self.model_name_or_path,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(self.chunk_size)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        image: Image.Image,
        mesh_resolution: int,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
    ) -> trimesh.Trimesh:
        LOGGER.info("Running TripoSR inference at marching cubes resolution %s", mesh_resolution)
        rgb = image.convert("RGB")
        with torch.no_grad():
            scene_codes = self.model([rgb], device=self.device)
            meshes = self.model.extract_mesh(scene_codes, True, resolution=int(mesh_resolution))
        mesh = meshes[0]
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        return mesh

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        return device


@dataclass
class Hunyuan3DBackend:
    model_name_or_path: str = "tencent/Hunyuan3D-2.1"
    repo_path: str = "vendor/Hunyuan3D-2.1"
    subfolder: str = "hunyuan3d-dit-v2-1"
    device: str = "auto"
    chunk_size: int = 8192

    def __post_init__(self) -> None:
        self.device = TripoSRBackend._resolve_device(self.device)
        self._ensure_import_path()
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Hunyuan3D-2.1 is not importable. Clone the repo into vendor\\Hunyuan3D-2.1 "
                "and install its requirements as shown in README.md."
            ) from exc

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        LOGGER.info("Loading Hunyuan3D model '%s' on %s", self.model_name_or_path, self.device)
        try:
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_name_or_path,
                subfolder=self.subfolder,
                device=self.device,
                dtype=dtype,
            )
        except TypeError:
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_name_or_path,
                subfolder=self.subfolder,
            )
            if hasattr(self.pipeline, "to"):
                self.pipeline.to(self.device)

    def generate(
        self,
        image: Image.Image,
        mesh_resolution: int,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
    ) -> trimesh.Trimesh:
        LOGGER.info(
            "Running Hunyuan3D inference at octree resolution %s, steps %s, guidance %.2f",
            mesh_resolution,
            num_inference_steps,
            guidance_scale,
        )
        with torch.no_grad():
            result = self.pipeline(
                image=image.convert("RGBA"),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                octree_resolution=int(mesh_resolution),
                num_chunks=int(self.chunk_size),
                output_type="trimesh",
            )
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        return mesh

    def _ensure_import_path(self) -> None:
        repo = Path(self.repo_path)
        if not repo.is_absolute():
            repo = Path.cwd() / repo
        for path in (repo, repo / "hy3dshape", repo / "hy3dpaint"):
            path_string = str(path)
            if path.exists() and path_string not in sys.path:
                sys.path.insert(0, path_string)
