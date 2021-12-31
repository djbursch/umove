from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse

from movenet.process_data import process_vid
from movenet.get_model import Model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Model()
input_size = model.input_size

@app.post("/upload_vid/")
def upload_video(user_id: str = Form(...), vid: UploadFile = File(...)):
    updated_vid = process_vid(user_id, vid, model, input_size)
    return Response(updated_vid, media_type='video/mp4')
    #return RedirectResponse("http://localhost:8000/show_vid/"+updated_vid)

@app.get("/upload_form/", response_class=HTMLResponse)
def upload_form_view(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})        

@app.get("/show_vid/{vid_name}", response_class=HTMLResponse)
def show_vid_view(request: Request, vid_name = str):
    vid_path = "/static/" + vid_name
    return templates.TemplateResponse("play_video.html", {"request": request, "video":vid_path})   

