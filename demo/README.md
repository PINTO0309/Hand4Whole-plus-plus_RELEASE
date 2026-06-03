# Hand4Whole++ Demo

This demo runs Hand4Whole++ inference on still images, an MP4/video file, or a USB camera.

## Required Files

Place these files under `demo/`:

- `deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx`
- `IH26M+ReIH+ARCTIC/snapshot_6.pth`
- `IH26M+ReIH+ARCTIC+AGORA/snapshot_6.pth`

Still images are read from `demo/inputs` by default, and outputs are written to `demo/outputs`.

## Still Images

Run the default still-image demo:

```bash
python demo/demo.py
```

Or specify an image directory:

```bash
python demo/demo.py --image-dir demo/inputs
```

Still-image mode saves:

- `*.obj`: SMPL-X mesh with hand colors
- `*_render_cropped_img.jpg`: render on the cropped model input
- `*_render_original_img.jpg`: render on the original image
- `*_smplx_param.json`: SMPL-X parameters

## Video File

Run inference on a video file:

```bash
python demo/demo.py --video-path sample.mp4 --output-video demo/outputs/sample_render.mp4
```

Video mode saves only the rendered MP4. If no body is detected in a frame, the original frame is written unchanged.

## USB Camera

Run inference from a USB camera:

```bash
python demo/demo.py --camera-id 0 --output-video demo/outputs/camera_0_render.mp4
```

Camera mode shows an OpenCV preview window and saves the rendered MP4. Stop the process with `q`, Esc, or `Ctrl+C`.

## Snapshot Selection

Choose the model snapshot with `--snapshot`:

```bash
python demo/demo.py --snapshot IH26M+ReIH+ARCTIC
python demo/demo.py --snapshot IH26M+ReIH+ARCTIC+AGORA
```

The default is `IH26M+ReIH+ARCTIC`.

## Output Directory

Use `--output-dir` to change where outputs are written:

```bash
python demo/demo.py --image-dir demo/inputs --output-dir demo/outputs_custom
```

For video and camera mode, `--output-video` controls the exact MP4 path. If omitted, the demo writes to:

- `demo/outputs/<video_name>_render.mp4` for video files
- `demo/outputs/camera_<id>_render.mp4` for USB cameras

## CLI Help

```bash
python demo/demo.py --help
```
