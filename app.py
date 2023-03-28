import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from generate import Generate, GenerateInput
from meta import Meta


app = FastAPI()
geneate : Generate
meta_config : Meta
logger = getLogger('uvicorn')


@app.on_event("startup")
def startup_event():
    global generate
    global meta_config

    cuda_env = os.getenv("ENABLE_CUDA")
    cuda_support=False
    cuda_core=""

    if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
        cuda_support=True
        cuda_core = os.getenv("CUDA_CORE")
        if cuda_core is None or cuda_core == "":
            cuda_core = "cuda:0"
        logger.info(f"CUDA_CORE set to {cuda_core}")
    else:
        logger.info("Running on CPU")

    model_dir = './models/model'

    generate = Generate(model_dir, cuda_support, cuda_core)
    meta_config = Meta(model_dir)


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta():
    return meta_config.get()


@app.post("/answers/")
async def read_item(item: GenerateInput, response: Response):
    try:
        response = await generate.do(item)
        return {
            "prompt": item.prompt,
            "response": response,
        }
    except Exception as e:
        logger.exception(
            f"Something went wrong while vectorizing data for prompt: {item.prompt}"
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}