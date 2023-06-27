import torch
import numpy as np
import bentoml
from bentoml.io import JSON
import helpers
import cv2
import time

class VehicleRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model_vehicle = torch.hub.load('ultralytics/yolov5', "custom", 'resources/weights/yolov5s.pt')
        self.model_plate = torch.hub.load('ultralytics/yolov5', "custom", 'resources/weights/last_plate_s_04042023_256.pt')
        self.model_digit = torch.hub.load('ultralytics/yolov5', "custom", 'resources/weights/last_digit_256_05052023_au.pt')
        self.list_model = [self.model_vehicle, self.model_plate, self.model_digit]
        if torch.cuda.is_available():
            for model in self.list_model:
                model.cuda()
        else:
            for model in self.list_model:
                model.cpu()

        # Config inference settings
        self.inference_size_vehicle = 640
        self.inference_size_plate = 256
        self.inference_size_digit = 256

        # Optional configs
        self.model_vehicle.classes = [2, 3, 5, 7]
        self.model_vehicle.conf = 0.5
        self.model_plate.conf = 0.4
        self.model_digit.conf = 0.6
        self.model_digit.agnostic_nms = True
        
        print("Finish init model")

    def vehicle_detection(self, input_img):
        # Return predictions only
        bboxes = []
        bboxes_dict = helpers.get_bboxes(self.model_vehicle, [input_img], size=self.inference_size_vehicle)
        if len(bboxes_dict) > 0:
            bboxes = bboxes_dict[0]
        return bboxes
    
    def plate_detection(self, images):
        bboxes_plate_list = helpers.get_bboxes(self.model_plate, images, size=self.inference_size_plate)
        return bboxes_plate_list
    
    def digit_detection(self, images):
        bboxes_digit_list = helpers.get_bboxes(self.model_digit, images, size=self.inference_size_digit)
        return bboxes_digit_list
    
    @bentoml.Runnable.method(batchable=False)
    def inference(self, payload):
        input_img_b64 = payload.get("image_base64")
        if input_img_b64 is None:
            return {"result": [], "image": ""}
        input_img = helpers.base64_2_img(input_img_b64)
        # inference_image = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2RGB)
        inference_image = input_img.copy()
        # inference_image = input_img.copy()
        result = {}
        infos = []
        infos_no_plate = []
        t_vehice_detection = time.perf_counter()
        bboxes_vehicle = self.vehicle_detection(inference_image)
        end_vehice_detection = time.perf_counter() - t_vehice_detection
        t_process_plate = time.perf_counter()
        if bboxes_vehicle:
            images = []
            for bbox in bboxes_vehicle:
                x1, y1, x2, y2, name = bbox
                crop = inference_image[y1:y2, x1:x2]
                images.append(crop)
            end_process_plate = time.perf_counter() - t_process_plate
            t_plate_detection = time.perf_counter()
            bboxes_plate_list = self.plate_detection(images)
            end_plate_detection = time.perf_counter() - t_plate_detection
            t_process_digit = time.perf_counter()
            for index, bboxes in enumerate(bboxes_plate_list):
                if bboxes:
                    vehicle_type = bboxes_vehicle[index][4]
                    if vehicle_type != "motorcycle" and len(bboxes) > 1:
                        # infos_no_plate.append({"vehicle": bboxes_vehicle[index], "digit": ""})
                        # continue
                        bboxes = helpers.filter_bboxes(index, bboxes, bboxes_vehicle)
                        bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)
                    if not bboxes:
                        infos_no_plate.append({"vehicle": bboxes_vehicle[index], "digit": ""})
                        continue
                    x1, y1, x2, y2 = helpers.expand_box(bboxes_vehicle[index][:4], bboxes[0][:4])
                    infos.append({"vehicle": bboxes_vehicle[index], "plate": helpers.expand_bbox_plate([x1, y1, x2, y2])})
                    bboxes_vehicle[index] += [x1, y1, x2, y2]
                else:
                    infos_no_plate.append({"vehicle": bboxes_vehicle[index], "digit": ""})
            images_digit = []
            for idx, info in enumerate(infos):
                x1, y1, x2, y2 = info["plate"]
                crop = inference_image[y1:y2, x1:x2]
                images_digit.append(crop)
            end_process_digit = time.perf_counter() - t_process_digit
            t_digit_detection = time.perf_counter()
            bboxes_digit_list = self.digit_detection(images_digit)
            end_digit_detection = time.perf_counter() - t_digit_detection
            t_result = time.perf_counter()
            for index, bboxes in enumerate(bboxes_digit_list):
                if bboxes:
                    x1, y1, x2, y2 = infos[index]["plate"]
                    crop = inference_image[y1:y2, x1:x2]
                    digit = helpers.handle_digit(crop, bboxes)
                    infos[index]["digit"] = digit
                else:
                    infos[index]["digit"] = ""
            result = infos + infos_no_plate
            result = sorted(result, key=lambda x: x["vehicle"][3], reverse=True)
            for info in result:
                x1, y1, x2, y2 = info["vehicle"][:4]
                digit = info["digit"]
                cv2.rectangle(input_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if info.get("plate") is not None:
                    x1_plate, y1_plate, x2_plate, y2_plate = info["plate"]
                    cv2.rectangle(input_img, (x1_plate, y1_plate), (x2_plate, y2_plate), (0, 0, 255), 2)
                    #text scale from number of character
                    scale = 0.7
                    #fill black rectangle
                    cv2.rectangle(input_img, (x1_plate, y1_plate - 30), (x1_plate + 15 * len(digit), y1_plate), (0, 0, 0), -1)
                    cv2.putText(input_img, digit, (x1_plate, y1_plate - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
            end_result = time.perf_counter() - t_result
            result_base64 = helpers.img_2_base64(input_img)                
            result_dict = {"result": result, "image": result_base64, "time": {"vehicle_detection": end_vehice_detection, "process_plate": end_process_plate, "plate_detection": end_plate_detection, "process_digit": end_process_digit, "digit_detection": end_digit_detection, "result": end_result}}
            return result_dict
        return {"result": [], "image": helpers.img_2_base64(input_img)}
    

yolo_v5_runner = bentoml.Runner(VehicleRunnable)

svc = bentoml.Service('vehicle-demo', runners=[ yolo_v5_runner ])

@svc.api(input=JSON(), output=JSON())
def inference(payload: dict):
    return yolo_v5_runner.inference.run(payload)