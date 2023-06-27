import cv2
import numpy as np
import base64
from torchvision import transforms
import time

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

viz_map = [
    (0, 0, 0), # Background.
    (255, 0, 0), # Flood.
]

list_points = {}


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


def preprocess(image, device):
    # transform the image to tensor and load into computation device
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    return image


def get_list_point(outputs):
    labels = np.argmax(outputs.squeeze(), axis=0)
    dict_points = get_position(labels)
    return dict_points

def get_position(matrix):
    # i = 0
    # value = 1
    # dict_points = {}
    matrix = np.random.randint(0, 2, size=(512, 512))
    return np.where(matrix == 1)