import bentoml
import helpers
import typing as t
import time

# triton runner
triton_runner = bentoml.triton.Runner(
    "triton_runner",
    "./model_repository",
    cli_args=["--model-control-mode=explicit", "--load-model=segment"],
)

svc = bentoml.Service(
    "triton-segment-tensorrt", runners=[triton_runner]
)


@svc.api(
    input=bentoml.io.JSON(), output=bentoml.io.JSON()
)
async def segment(payload: dict) -> dict:
    t = time.perf_counter()
    base64_str = payload["image_base64"]
    image = helpers.base64_2_cv2(base64_str)
    image_preprocessed = helpers.preprocess(image, "cuda") 
    time_preprocess = time.perf_counter() - t
    t = time.perf_counter()
    InferResult = await triton_runner.segment.async_run(image_preprocessed.cpu().numpy())
    output = InferResult.as_numpy("outputs")
    time_inference = time.perf_counter() - t
    t = time.perf_counter()
    list_point  = helpers.get_list_point(output)
    time_postprocess = time.perf_counter() - t
    results = {"list_point": list_point, "time_preprocess": time_preprocess, "time_inference": time_inference, "time_postprocess": time_postprocess}
    return results
    
    
# Triton Model management API
@svc.api(
    input=bentoml.io.JSON.from_sample({"model_name": "segment"}),
    output=bentoml.io.JSON(),
)
async def model_config(input_model: dict):
    return await triton_runner.get_model_config(input_model["model_name"], as_json=True)


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
async def list_models(_) -> list:
    resp = await triton_runner.get_model_repository_index()
    return [i.name for i in resp.models]