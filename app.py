from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image

from src.background import remove_background
from src.export import export_mesh, export_preview_mesh
from src.inference import Hunyuan3DBackend, ImageToMeshBackend, TripoSRBackend
from src.mesh_utils import clean_mesh, load_mesh_summary, prepare_preview_mesh, strip_visuals
from src.preprocess import crop_recenter_pad, rgba_to_triposr_rgb


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
TEMP_DIR = ROOT / "temp"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

LOGGER = logging.getLogger("toybox_photo_to_obj")

backend: Optional[ImageToMeshBackend] = None
backend_cache_key: Optional[tuple[str, str, str, str, str, int]] = None


def get_backend(
    backend_name: str,
    device: str,
    model_name_or_path: str,
    hunyuan_repo_path: str,
    hunyuan_subfolder: str,
    chunk_size: int,
) -> ImageToMeshBackend:
    global backend, backend_cache_key
    key = (backend_name, device, model_name_or_path, hunyuan_repo_path, hunyuan_subfolder, chunk_size)
    if backend is None or backend_cache_key != key:
        if backend_name == "hunyuan3d":
            backend = Hunyuan3DBackend(
                model_name_or_path=model_name_or_path,
                repo_path=hunyuan_repo_path,
                subfolder=hunyuan_subfolder,
                device=device,
                chunk_size=chunk_size,
            )
        elif backend_name == "triposr":
            backend = TripoSRBackend(
                model_name_or_path=model_name_or_path,
                device=device,
                chunk_size=chunk_size,
            )
        else:
            raise gr.Error(f"Unknown backend: {backend_name}")
        backend_cache_key = key
    return backend


def process_image(input_image: Image.Image, foreground_ratio: float) -> Tuple[Image.Image, Image.Image, Path]:
    if input_image is None:
        raise gr.Error("Upload an image first.")

    rgba = remove_background(input_image)
    centered = crop_recenter_pad(rgba, foreground_ratio=foreground_ratio)
    triposr_rgb = rgba_to_triposr_rgb(centered)

    processed_path = TEMP_DIR / "processed.png"
    triposr_rgb.save(processed_path)
    return centered, triposr_rgb, processed_path


def generate_mesh(
    input_image: Image.Image,
    backend_name: str,
    export_stl: bool,
    foreground_ratio: float,
    mesh_resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
    triposr_model: str,
    hunyuan_model: str,
    hunyuan_repo_path: str,
    hunyuan_subfolder: str,
    chunk_size: int,
) -> Tuple[Image.Image, str, str, Optional[str], str]:
    centered_rgba, triposr_image, processed_path = process_image(input_image, foreground_ratio)

    model_name_or_path = hunyuan_model if backend_name == "hunyuan3d" else triposr_model
    mesh_backend = get_backend(
        backend_name=backend_name,
        device=device,
        model_name_or_path=model_name_or_path,
        hunyuan_repo_path=hunyuan_repo_path,
        hunyuan_subfolder=hunyuan_subfolder,
        chunk_size=chunk_size,
    )
    inference_image = centered_rgba if backend_name == "hunyuan3d" else triposr_image
    raw_mesh = mesh_backend.generate(
        inference_image,
        mesh_resolution=mesh_resolution,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    mesh = clean_mesh(raw_mesh)
    exportable_mesh = strip_visuals(mesh)

    obj_path, stl_path = export_mesh(exportable_mesh, OUTPUT_DIR, export_stl=export_stl)
    preview_path = export_preview_mesh(prepare_preview_mesh(exportable_mesh), TEMP_DIR)
    summary = load_mesh_summary(exportable_mesh, obj_path=obj_path, stl_path=stl_path, processed_path=processed_path)

    return centered_rgba, str(preview_path), str(obj_path), str(stl_path) if stl_path else None, summary


def build_ui(args: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(title="Toybox Photo to OBJ") as demo:
        gr.Markdown(
            "# Toybox Photo to OBJ\n"
            "Local MVP for turning one object photo into an OBJ mesh. First run downloads models; later runs are local."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Object Photo",
                    type="pil",
                    image_mode="RGBA",
                    sources=["upload"],
                )
                generate_button = gr.Button("Generate OBJ", variant="primary")

                with gr.Accordion("Settings", open=False):
                    backend_name = gr.Dropdown(
                        label="Mesh Backend",
                        choices=[
                            ("Hunyuan3D-2.1", "hunyuan3d"),
                            ("TripoSR", "triposr"),
                        ],
                        value=args.backend,
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Size",
                        minimum=0.5,
                        maximum=0.95,
                        value=0.85,
                        step=0.05,
                    )
                    mesh_resolution = gr.Slider(
                        label="Mesh Resolution",
                        minimum=64,
                        maximum=args.max_resolution,
                        value=args.mesh_resolution,
                        step=32,
                    )
                    num_inference_steps = gr.Slider(
                        label="Hunyuan Steps",
                        minimum=5,
                        maximum=100,
                        value=args.steps,
                        step=5,
                    )
                    guidance_scale = gr.Slider(
                        label="Hunyuan Guidance",
                        minimum=1.0,
                        maximum=10.0,
                        value=args.guidance,
                        step=0.5,
                    )
                    export_stl = gr.Checkbox(label="Also export STL", value=True)

            with gr.Column(scale=1):
                processed_image = gr.Image(label="Background Removed / Centered", type="pil")
                mesh_preview = gr.Model3D(label="Mesh Preview", height=420, clear_color=(1, 1, 1, 1))

        with gr.Row():
            obj_file = gr.File(label="Download OBJ")
            stl_file = gr.File(label="Download STL")

        status = gr.Textbox(label="Status", lines=8)

        generate_button.click(
            fn=generate_mesh,
            inputs=[
                input_image,
                backend_name,
                export_stl,
                foreground_ratio,
                mesh_resolution,
                num_inference_steps,
                guidance_scale,
                gr.State(args.device),
                gr.State(args.triposr_model),
                gr.State(args.hunyuan_model),
                gr.State(args.hunyuan_repo),
                gr.State(args.hunyuan_subfolder),
                gr.State(args.chunk_size),
            ],
            outputs=[processed_image, mesh_preview, obj_file, stl_file, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local photo-to-OBJ MVP for Toybox import.")
    parser.add_argument("--port", type=int, default=7860, help="Local Gradio port.")
    parser.add_argument("--listen", action="store_true", help="Listen on 0.0.0.0 instead of localhost.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--backend", choices=["hunyuan3d", "triposr"], default="hunyuan3d", help="Mesh generation backend.")
    parser.add_argument("--triposr-model", default="stabilityai/TripoSR", help="TripoSR Hugging Face model id or local path.")
    parser.add_argument("--hunyuan-model", default="tencent/Hunyuan3D-2.1", help="Hunyuan3D Hugging Face model id or local path.")
    parser.add_argument("--hunyuan-repo", default="vendor/Hunyuan3D-2.1", help="Local Hunyuan3D-2.1 source checkout.")
    parser.add_argument("--hunyuan-subfolder", default="hunyuan3d-dit-v2-1", help="Hunyuan3D shape model subfolder.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Renderer/inference chunk size.")
    parser.add_argument("--mesh-resolution", type=int, default=384, help="Default mesh resolution.")
    parser.add_argument("--max-resolution", type=int, default=512, help="Maximum UI mesh resolution.")
    parser.add_argument("--steps", type=int, default=30, help="Hunyuan3D inference steps.")
    parser.add_argument("--guidance", type=float, default=5.0, help="Hunyuan3D guidance scale.")
    parser.add_argument("--model", dest="triposr_model", help=argparse.SUPPRESS)
    parser.add_argument("--mc-resolution", dest="mesh_resolution", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    demo = build_ui(args)
    demo.queue(max_size=1)
    demo.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
