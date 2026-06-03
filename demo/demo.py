from __future__ import annotations

import sys
import os
import os.path as osp
import argparse
from typing import Any, TypeAlias, TypedDict, cast

demo_dir = osp.dirname(osp.abspath(__file__))
root_dir = osp.abspath(osp.join(demo_dir, '..'))
SNAPSHOT_PATHS = {
    'IH26M+ReIH+ARCTIC': osp.join('IH26M+ReIH+ARCTIC', 'snapshot_6.pth'),
    'IH26M+ReIH+ARCTIC+AGORA': osp.join('IH26M+ReIH+ARCTIC+AGORA', 'snapshot_6.pth'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--image-dir',
        default=None,
        help='Directory of still images to process. Defaults to demo/inputs when no input source is specified.',
    )
    input_group.add_argument(
        '--video-path',
        default=None,
        help='Path to an input video file.',
    )
    input_group.add_argument(
        '--camera-id',
        type=int,
        default=None,
        help='USB camera index to process.',
    )
    parser.add_argument(
        '--output-dir',
        default=osp.join(demo_dir, 'outputs'),
        help='Directory for output files.',
    )
    parser.add_argument(
        '--output-video',
        default=None,
        help='Output mp4 path for video or camera mode.',
    )
    parser.add_argument(
        '--snapshot',
        default='IH26M+ReIH+ARCTIC',
        choices=tuple(SNAPSHOT_PATHS.keys()),
        help='Snapshot preset to load.',
    )
    return parser.parse_args()


args = parse_args()
sys.path.insert(0, osp.join(root_dir, 'main'))

import numpy as np
import numpy.typing as npt
import cv2
import json
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

from config import cfg
from model import get_model
from utils.preprocessing import set_aspect_ratio, get_patch_img
from utils.smpl_x import smpl_x
from utils.vis import render_mesh
from glob import glob
from tqdm import tqdm

FloatArray: TypeAlias = npt.NDArray[np.float32]
UInt8Array: TypeAlias = npt.NDArray[np.uint8]
IntArray: TypeAlias = npt.NDArray[np.int64]
BBox: TypeAlias = list[float]
Color: TypeAlias = tuple[float, float, float]
DetectorOutput: TypeAlias = npt.NDArray[np.float32]


class ModelOutput(TypedDict):
    smplx_vert_cam: torch.Tensor
    smplx_root_pose: torch.Tensor
    smplx_body_pose: torch.Tensor
    smplx_lhand_pose: torch.Tensor
    smplx_rhand_pose: torch.Tensor
    smplx_jaw_pose: torch.Tensor
    smplx_shape: torch.Tensor
    smplx_expr: torch.Tensor


DETECTOR_INPUT_SHAPE = (640, 640)
BODY_CLASS_ID = 0
BODY_DETECTION_SCORE_THRESHOLD = 0.25


class LetterboxInfo(TypedDict):
    scale: float
    pad_x: float
    pad_y: float
    original_width: int
    original_height: int


def prepare_detector_input(rgb_img: UInt8Array) -> tuple[FloatArray, LetterboxInfo]:
    input_height, input_width = DETECTOR_INPUT_SHAPE
    original_height, original_width = rgb_img.shape[:2]
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    pad_x = (input_width - resized_width) / 2.0
    pad_y = (input_height - resized_height) / 2.0

    resized_img = cast(UInt8Array, cv2.resize(cast(Any, rgb_img), (resized_width, resized_height), interpolation=cv2.INTER_LINEAR))
    input_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    input_img[top:top + resized_height, left:left + resized_width] = resized_img
    input_tensor = np.divide(input_img.astype(np.float32), 255.0, dtype=np.float32)
    input_tensor = input_tensor.transpose(2, 0, 1)[None]

    return cast(FloatArray, input_tensor), {
        'scale': scale,
        'pad_x': float(left),
        'pad_y': float(top),
        'original_width': original_width,
        'original_height': original_height,
    }


def detector_xyxy_to_original_xywh(xyxy: npt.NDArray[Any], info: LetterboxInfo) -> BBox:
    input_height, input_width = DETECTOR_INPUT_SHAPE
    coords = np.array(xyxy, dtype=np.float32).reshape(4)
    x1, y1, x2, y2 = [float(v) for v in coords]
    if float(np.nanmax(coords)) <= 2.0:
        x1 *= input_width
        x2 *= input_width
        y1 *= input_height
        y2 *= input_height

    x1 = float(np.clip((x1 - info['pad_x']) / info['scale'], 0.0, float(info['original_width'] - 1)))
    x2 = float(np.clip((x2 - info['pad_x']) / info['scale'], 0.0, float(info['original_width'] - 1)))
    y1 = float(np.clip((y1 - info['pad_y']) / info['scale'], 0.0, float(info['original_height'] - 1)))
    y2 = float(np.clip((y2 - info['pad_y']) / info['scale'], 0.0, float(info['original_height'] - 1)))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def get_body_box_from_detector_output(output: DetectorOutput, info: LetterboxInfo) -> BBox | None:
    detections = output[0] if output.ndim == 3 else output
    if detections.size == 0:
        return None

    best_idx: int | None = None
    best_score = BODY_DETECTION_SCORE_THRESHOLD
    for idx, detection in enumerate(detections):
        class_id = int(detection[0])
        score = float(detection[5])
        if class_id == BODY_CLASS_ID and score >= best_score:
            best_idx = idx
            best_score = score
    if best_idx is None:
        return None

    return detector_xyxy_to_original_xywh(detections[best_idx, 1:5], info)


def read_rgb_image(img_path: str) -> UInt8Array:
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        raise IOError('Fail to read {}'.format(img_path))
    return cast(UInt8Array, cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))


def bgr_to_rgb_frame(frame: npt.NDArray[Any]) -> UInt8Array:
    return cast(UInt8Array, cv2.cvtColor(cast(Any, frame), cv2.COLOR_BGR2RGB))


def rgb_to_bgr_frame(frame: npt.NDArray[Any]) -> UInt8Array:
    return cast(UInt8Array, cv2.cvtColor(cast(Any, frame), cv2.COLOR_RGB2BGR))


def scaled_color(color: Color, scale: float) -> FloatArray:
    return np.array([c * scale for c in color], dtype=np.float32).reshape(1,3)


def tensor_batch_item_to_numpy(tensor: torch.Tensor) -> FloatArray:
    return cast(FloatArray, tensor.detach().cpu().numpy()[0])


def flatten_float_array(array: npt.NDArray[Any]) -> list[float]:
    return cast(list[float], array.reshape(-1).tolist())


def save_obj_w_color(v: FloatArray, f: IntArray, color: FloatArray | None = None, file_name: str = 'output.obj') -> None:
    with open(file_name, 'w') as obj_file:
        for i in range(len(v)):
            if color is None:
                obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
            else:
                obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + ' ' + str(color[i][0]) + ' ' + str(color[i][1]) + ' ' + str(color[i][2]) + '\n')
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '\n')


def save_smplx_params(out: ModelOutput, file_name: str) -> None:
    root_pose = tensor_batch_item_to_numpy(out['smplx_root_pose'])
    body_pose = tensor_batch_item_to_numpy(out['smplx_body_pose'])
    lhand_pose = tensor_batch_item_to_numpy(out['smplx_lhand_pose'])
    rhand_pose = tensor_batch_item_to_numpy(out['smplx_rhand_pose'])
    jaw_pose = tensor_batch_item_to_numpy(out['smplx_jaw_pose'])
    shape = tensor_batch_item_to_numpy(out['smplx_shape'])
    expr = tensor_batch_item_to_numpy(out['smplx_expr'])
    with open(file_name, 'w') as f:
        json.dump({'root_pose': flatten_float_array(root_pose), \
                'body_pose': flatten_float_array(body_pose), \
                'lhand_pose': flatten_float_array(lhand_pose), \
                'rhand_pose': flatten_float_array(rhand_pose), \
                'jaw_pose': flatten_float_array(jaw_pose), \
                'shape': flatten_float_array(shape), \
                'expr': flatten_float_array(expr)}, f)


def process_frame(original_img: UInt8Array, frame_name: str, save_static_outputs: bool) -> UInt8Array | None:
    detector_input, letterbox_info = prepare_detector_input(original_img)
    detector_output = cast(DetectorOutput, detector.run([detector_output_name], {detector_input_name: detector_input})[0])
    person_bbox = get_body_box_from_detector_output(detector_output, letterbox_info)
    if person_bbox is None:
        return None

    bbox = cast(FloatArray, set_aspect_ratio(person_bbox, cfg.input_img_shape[1]/cfg.input_img_shape[0]))
    patch_img, _img2bb_trans, _bb2img_trans = get_patch_img(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
    patch_img = cast(FloatArray, patch_img)
    img = transform(patch_img.astype(np.float32)).div(255.0)
    img = img.cuda()[None,:,:,:]

    inputs: dict[str, torch.Tensor] = {'img': img}
    targets: dict[str, Any] = {}
    meta_info: dict[str, Any] = {}
    with torch.no_grad():
        out = cast(ModelOutput, model(inputs, targets, meta_info, 'test'))
    vert = tensor_batch_item_to_numpy(out['smplx_vert_cam'])

    if save_static_outputs:
        color = np.full((smpl_x.vertex_num,3), 0.8, dtype=np.float32)
        color[smpl_x.hand_vertex_idx['right_hand'],:] = scaled_color(rhand_color, 0.8)
        color[smpl_x.hand_vertex_idx['left_hand'],:] = scaled_color(lhand_color, 0.8)
        save_obj_w_color(vert, smpl_x.face, color, osp.join(output_root_path, frame_name + '.obj'))

        vis_img = img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
        focal = [cfg.focal[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.focal[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * cfg.input_img_shape[1], cfg.princpt[1] / cfg.input_body_shape[0] * cfg.input_img_shape[0]]
        rendered_cropped_img = render_mesh(vert, smpl_x.face, {'focal': focal, 'princpt': princpt}, vis_img)[:,:,::-1]
        cv2.imwrite(osp.join(output_root_path, frame_name + '_render_cropped_img.jpg'), rendered_cropped_img)

        save_smplx_params(out, osp.join(output_root_path, frame_name + '_smplx_param.json'))

    vis_img = original_img.copy()
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    render_color = torch.ones((1,smpl_x.vertex_num,3)).float().cuda()
    render_color[:,smpl_x.hand_vertex_idx['right_hand'],:] = torch.FloatTensor(rhand_color).cuda()[None,:]
    render_color[:,smpl_x.hand_vertex_idx['left_hand'],:] = torch.FloatTensor(lhand_color).cuda()[None,:]
    rendered_img = render_mesh(vert, smpl_x.face, {'focal': focal, 'princpt': princpt}, vis_img, color=render_color)[:,:,::-1]

    if save_static_outputs:
        cv2.imwrite(osp.join(output_root_path, frame_name + '_render_original_img.jpg'), rendered_img)

    return cast(UInt8Array, rendered_img)


def run_image_dir(image_dir: str) -> None:
    img_path_list: list[str] = glob(osp.join(image_dir, '*'))
    for img_path in tqdm(img_path_list):
        frame_name = osp.splitext(osp.basename(img_path))[0]
        try:
            original_img = read_rgb_image(img_path)
        except IOError as e:
            print(str(e) + '; skipping.')
            continue
        rendered_img = process_frame(original_img, frame_name, True)
        if rendered_img is None:
            print('No body detected in {}; skipping.'.format(img_path))


def get_default_output_video_path(video_path: str | None, camera_id: int | None) -> str:
    if video_path is not None:
        video_name = osp.splitext(osp.basename(video_path))[0]
        return osp.join(output_root_path, video_name + '_render.mp4')
    camera_name = 'camera_{}'.format(camera_id if camera_id is not None else 0)
    return osp.join(output_root_path, camera_name + '_render.mp4')


def get_capture_fps(capture: Any) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or fps != fps:
        return 30.0
    return fps


def create_video_writer(output_video_path: str, fps: float, frame_width: int, frame_height: int) -> Any:
    os.makedirs(osp.dirname(osp.abspath(output_video_path)), exist_ok=True)
    fourcc = cast(Any, cv2).VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise IOError('Fail to open video writer at {}'.format(output_video_path))
    return writer


def run_capture(capture: Any, output_video_path: str, source_name: str, show_gui: bool = False) -> None:
    fps = get_capture_fps(capture)
    writer: Any | None = None
    frame_idx = 0
    window_name = 'Hand4Whole++ Demo'
    try:
        with tqdm(desc=source_name, unit='frame') as progress:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame = cast(UInt8Array, frame)
                if writer is None:
                    frame_height, frame_width = frame.shape[:2]
                    writer = create_video_writer(output_video_path, fps, int(frame_width), int(frame_height))
                assert writer is not None

                original_img = bgr_to_rgb_frame(frame)
                rendered_img = process_frame(original_img, '{:06d}'.format(frame_idx), False)
                if rendered_img is None:
                    output_frame = frame
                else:
                    output_frame = rendered_img
                writer.write(output_frame)
                if show_gui:
                    cv2.imshow(window_name, cast(Any, output_frame))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                frame_idx += 1
                progress.update(1)
    except KeyboardInterrupt:
        print('Interrupted; wrote {} frames to {}'.format(frame_idx, output_video_path))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show_gui:
            cv2.destroyWindow(window_name)


def run_video(video_path: str, output_video_path: str) -> None:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError('Fail to open video {}'.format(video_path))
    run_capture(capture, output_video_path, osp.basename(video_path))


def run_camera(camera_id: int, output_video_path: str) -> None:
    capture = cv2.VideoCapture(camera_id)
    if not capture.isOpened():
        raise IOError('Fail to open camera {}'.format(camera_id))
    run_capture(capture, output_video_path, 'camera_{}'.format(camera_id), show_gui=True)


root_path = demo_dir
input_root_path = cast(str, args.image_dir) if args.image_dir is not None else osp.join(root_path, 'inputs')
output_root_path = osp.abspath(cast(str, args.output_dir))
os.makedirs(output_root_path, exist_ok=True)
rhand_color: Color = (0.6, 0.7, 1.0)
lhand_color: Color = (0.7, 1.0, 0.7)


# snapshot load
snapshot_name = cast(str, args.snapshot)
model_path = osp.join(root_path, SNAPSHOT_PATHS[snapshot_name])
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
for module in model.module.trainable_modules+model.module.eval_modules:
    module.eval()
cudnn.benchmark = True

# body detector
detector_path = osp.join(root_path, 'deimv2_dinov3_x_wholebody49_ins_s08_maskhead256x3_center_1240query.onnx')
assert osp.exists(detector_path), 'Cannot find body detector at ' + detector_path
available_providers = ort.get_available_providers()
detector_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
detector = ort.InferenceSession(detector_path, providers=detector_providers)
detector_input_name = detector.get_inputs()[0].name
detector_output_name = detector.get_outputs()[0].name

transform = transforms.ToTensor()

video_path = cast(str | None, args.video_path)
camera_id = cast(int | None, args.camera_id)
output_video = cast(str | None, args.output_video)

if video_path is not None:
    run_video(video_path, output_video if output_video is not None else get_default_output_video_path(video_path, None))
elif camera_id is not None:
    run_camera(camera_id, output_video if output_video is not None else get_default_output_video_path(None, camera_id))
else:
    run_image_dir(input_root_path)
