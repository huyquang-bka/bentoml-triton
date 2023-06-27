import cv2
import numpy as np
import base64
import torch
from torchvision import transforms
import time


def base64_2_cv2(base64_str: str) -> np.ndarray:
    """
    Convert base64 string to cv2 image

    Args:
        base64_str (str): base64 string

    Returns:
        np.ndarray: cv2 image
    """
    img_bytes = base64.b64decode(base64_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
    return img

def cv2_2_base64(img: np.ndarray) -> str:
    """
    Convert cv2 image to base64 string

    Args:
        img (np.ndarray): cv2 image

    Returns:
        str: base64 string
    """
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = base64.b64encode(buffer)
    img_str = img_bytes.decode("utf-8")
    return img_str


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def pred_to_bboxes(image, pred):
    size = torch.Size([640, 640])
    for i, det in enumerate(pred):  # per image
        bboxes = []
        det = torch.from_numpy(det).squeeze(0)
        if len(det):
            print("det.shape: ", det.shape)
            det[:, :4] = scale_boxes(size, det[:, :4], image.shape).round()
            for bbox in det:
                bbox = bbox.tolist()
                x1, y1, x2, y2 = list(map(lambda x: max(0, int(x)), bbox[:4]))
                bboxes.append([x1, y1, x2, y2, bbox[4], bbox[5]])
    return bboxes
            