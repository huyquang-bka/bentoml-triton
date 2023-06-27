import bentoml
import helpers
import typing as t
import time
import numpy as np
import cv2

# triton runner
triton_runner = bentoml.triton.Runner(
    "triton_runner",
    "./model_repository"
)

svc = bentoml.Service(
    "triton-vehicle", runners=[triton_runner]
)


@svc.api(
    input=bentoml.io.JSON(), output=bentoml.io.JSON()
)
async def ensemble(payload: dict) -> dict:
    base64_str = payload["image_base64"]
    image = helpers.base64_2_cv2(base64_str)
    images = np.expand_dims(image, axis=0)
    InferResult = await triton_runner.ensemble.async_run(images)
    output = InferResult.as_numpy("output-post-vehicle")
    bboxes = helpers.pred_to_bboxes(image, output)
    for bbox in bboxes:
        x1, y1, x2, y2, conf, cls = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{cls}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    result_base64 = helpers.cv2_2_base64(image)
    return {"output": bboxes, "result_base64": result_base64}


@svc.api(
    input=bentoml.io.JSON(), output=bentoml.io.JSON()
)
async def pre_vehicle(payload: dict) -> dict:
    base64_str = payload["image_base64"]
    image = helpers.base64_2_cv2(base64_str)
    images = np.expand_dims(image, axis=0)
    InferResult = await triton_runner.pre_vehicle.async_run(images)
    output = InferResult.as_numpy("output-pre-vehicle")
    print("output.shape", output.shape)
    return {"output shape": output.shape}

    
# Triton Model management API

@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def list_models(_) -> list:
    resp = await triton_runner.get_model_repository_index()
    return [i.name for i in resp.models]