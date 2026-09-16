import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form,BackgroundTasks
from pathlib import Path
from starlette.middleware.cors import CORSMiddleware
import uuid
import prediction
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

BASE_DIR=Path(__file__).resolve().parent
UPLOAD_DIR=BASE_DIR/"uploads"
RESULT_DIR=BASE_DIR/"results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)



api=FastAPI(title="Brain Tumor Segmentation API",)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api.mount("/static", StaticFiles(directory=RESULT_DIR), name="static")
STEPS=[
    "load_file",
    "preprocessing",
    "running_UNet",
    "generating_segmentation"
]

jobs={}

def create_job():
    job_id=str(uuid.uuid4())
    jobs[job_id]={
        "steps":{step:"pending" for step in STEPS},
        "current_step":None,
        "error":None,
        "result_files":[]
    }
    return job_id

def update_step(job_id,step,status):
    jobs[job_id]["steps"][step]=status
    jobs[job_id]["current_step"]=step

def set_error(job_id,error):
    jobs[job_id]["error"]=error

def get_status(job_id):
    return jobs.get(job_id)

@api.post("/upload")
async def upload(background_tasks: BackgroundTasks,file:UploadFile = File(...),slice_number:int=Form(),):
    job_id=create_job()
    save_path=os.path.join(UPLOAD_DIR,f"f{job_id}_{slice_number}_{file.filename}")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_process_job,job_id,save_path,slice_number)

    return {"job_id": job_id, "filename": file.filename,"slice":slice_number}
async def run_process_job(job_id: str, filepath: str, slice_number: int):
    await run_in_threadpool(process_job, job_id, filepath, slice_number)
@api.get("/status/{job_id}")
def status(job_id):
    job=get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

def process_job(job_id:str,filepath:str,slice:int):
    try:
        if not os.path.exists(filepath):
            raise Exception("Фајлот не постои.")
        update_step(job_id,"load_file","done")
        volume=prediction.preprocessing(filepath,slice)
        update_step(job_id,"preprocessing","done")
        model=prediction.load_model()
        raw_output=prediction.predict(model,volume)
        update_step(job_id,"running_UNet","done")
        mask=prediction.postprocessing(raw_output)
        result_folder = os.path.join(
            RESULT_DIR,
            job_id
        )
        jobs[job_id]["result_files"]=prediction.exportImage(mask,volume,result_folder,job_id)
        update_step(job_id,"generating_segmentation","done")


    except Exception as e:
        set_error(job_id, str(e))

@api.post("/tumor")
def tumor(job_Id:str):
    path=os.path.join("result/",job_Id,"/mask.png")
