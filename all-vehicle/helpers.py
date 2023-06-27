import cv2
import numpy as np
import base64


def img_2_base64(img):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


def base64_2_img(base64_str):
    jpg_original = base64.b64decode(base64_str)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    return img


def get_bboxes(model, input_imgs, size=640):
    results = model(input_imgs, size=size)
    bboxes_list = []
    for idx, result in enumerate(results.pandas().xyxy):
        bboxes = []
        for index, data in result.iterrows():
            bbox = list(map(int, [data.xmin, data.ymin, data.xmax, data.ymax]))
            try:
                name = int(data["name"])
            except:
                name = data["name"]
            bbox.append(name)
            bboxes.append(bbox)
        bboxes_list.append(bboxes)
    return bboxes_list

def expand_box(box1, box2):
    x1, y1, x2, y2 = box1
    x1_plate, y1_plate, x2_plate, y2_plate = box2
    x1_plate += x1
    y1_plate += y1
    x2_plate += x1
    y2_plate += y1
    return [x1_plate, y1_plate, x2_plate, y2_plate]


def expand_bbox_plate(bbox, expand=3):
    return [bbox[0] - expand, bbox[1] - expand, bbox[2] + expand, bbox[3] + expand]

def check_square_plate(crop):
    return crop.shape[1] / crop.shape[0] < 2.5

def get_plate(is_square_plate, bboxes):
    if not is_square_plate:
        return "".join([str(bbox[4]) for bbox in bboxes])
    y_avarage = sum([bbox[1] for bbox in bboxes]) / len(bboxes)
    line_1 = sorted([bbox for bbox in bboxes if bbox[1] <= y_avarage], key=lambda x: x[0])
    line_2 = sorted([bbox for bbox in bboxes if bbox[1] > y_avarage], key=lambda x: x[0])
    plate_1 = "".join([str(bbox[4]) for bbox in line_1])
    plate_2 = "".join([str(bbox[4]) for bbox in line_2])
    return plate_1 + "-" + plate_2

def check_plate_valid(plate_text):
    try:
        int(plate_text)
        return False
    except:
        pass
    if 7 <= len(plate_text) <= 11:
        return True
    return False

def handle_digit(crop, bboxes):
    bboxes = sorted(bboxes, key=lambda x: x[0])
    is_square_plate = check_square_plate(crop)
    plate = get_plate(is_square_plate, bboxes)
    if check_plate_valid(plate):
        return plate.upper()
    return ""

def filter_bboxes(index, bboxes_plate, bboxes_vehicle):
    new_bboxes = []
    base_vehicle_bbox = bboxes_vehicle[index]
    for bbox_plate in bboxes_plate:
        count = 0
        is_append = True
        for bbox_vehicle in bboxes_vehicle:
            x_plate_center = (bbox_plate[0] + base_vehicle_bbox[0] + bbox_plate[2] + base_vehicle_bbox[0]) / 2
            y_plate_center = (bbox_plate[1] + base_vehicle_bbox[1] + bbox_plate[3] + base_vehicle_bbox[1]) / 2
            if bbox_vehicle[0] < x_plate_center < bbox_vehicle[2] and bbox_vehicle[1] < y_plate_center < bbox_vehicle[3]:
                count += 1
                if count == 2:
                    is_append = False
        if is_append:
            new_bboxes.append(bbox_plate)
    return new_bboxes