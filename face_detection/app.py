# fastapi packages
from fastapi import Request, FastAPI, File, Form
import uvicorn
from starlette.responses import Response

# rest of the packages
import cv2
import numpy as np
import io, os, time, json, base64
import requests
from datetime import datetime,timedelta

from face_detection_funs import *

image_dir = os.path.join(os.getcwd(),'static')
os.makedirs(image_dir, exist_ok=True)
input_image_path = os.path.join(image_dir,'input_image.png')
output_image_path = os.path.join(image_dir,'output_image.png')

main_dir = 'data'
os.makedirs(main_dir,exist_ok=True)

def save_image_locally(request_json):
    try:
        content=request_json["image"]
        print("type content =",type(content))
        if ";base64," in content:
            data=content.split(";base64,")[1].encode("utf8")
        else:
            data=content.encode("utf8")
        os.makedirs(main_dir,exist_ok=True)
        image_path="{}/{}".format(main_dir,"local_image.png")
        with open(image_path, "wb") as fp:
            fp.write(base64.decodebytes(data))
        return image_path
    except Exception as e:
        print("exception in save_images_locally =",e)

def convert_cv2_to_base64(annotated_frame):
    success, encoded_image1 = cv2.imencode('.png', annotated_frame)
    annotated_image=encoded_image1.tobytes()
    im_b64 = base64.b64encode(annotated_image).decode()
    return im_b64

app = FastAPI()

#### Code To handle get request
@app.get("/")
async def process_get_request():
    return "This API accept json request with input image & respond with faces detected."

@app.post("/face_detection")
async def process_post_json_data(request: Request):
    payload = await request.json()
    
    ''''''
    if not payload:
        msg = "no message received"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400

    current_time = str(datetime.now()+timedelta(hours=5,minutes=30))
    request_json=payload.copy()
    content = request_json["image"]
    algo = request_json["Algorithm"]
    
    #image_list=[request_json["image"]]
    #image_local_paths=save_images_locally(image_list)
    #content = image_list[0]

    if ";base64," in content:
        b64_img = content.split(";base64,")[1].encode("utf8")
    else:
        b64_img = content.encode("utf8")

    with open(input_image_path, "wb") as fp:
        fp.write(base64.decodebytes(b64_img))
    
    output_json = {}
    try:
        if algo == 'cascade_classifiers_face_detector':
            img, detections = cascade_classifiers_face_detector(input_image_path)

        elif algo == "dlib_hog_face_detector":
            img, detections = dlib_hog_face_detector(input_image_path)

        elif algo == "dlib_cnn_face_detector":
            img, detections = dlib_cnn_face_detector(input_image_path)

        elif algo == "ssd_face_detector":
            img, detections = ssd_face_detector(input_image_path)

        elif algo == "mtcnn_face_detector":
            img, detections = mtcnn_face_detector(input_image_path)

        elif algo == "face_detection_detector":
            img, detections = face_detection_detector(input_image_path)

        elif algo == "retina_face_detector":
            img, detections = retina_face_detector(input_image_path)

        elif algo == "mediapipe_face_detector":
            img, detections = mediapipe_face_detector(input_image_path)

        elif algo == "yunet_face_detector":
            img, detections = yunet_face_detector(input_image_path)

        elif algo == "facenet_pytorch_mtcnn_face_detector":
            img, detections = facenet_pytorch_mtcnn_face_detector(input_image_path)

        else:
            output_json = {"status":"failure", "message":"Invalid algorithm name" }
            print("Invalid algorithm name")

        print("detections =",detections)

        im_b64=convert_cv2_to_base64(img)
        if len(detections) != []:
#            output_json={"status":"success","message":"","timestamp":current_time,"result":{"classes":classIds,"scores":scores,"bboxes":boxes,"image":im_b64}}
            output_json={"status":"success","message":"","timestamp":current_time,"result":{"detections":detections,"image":im_b64}}
        else:
            output_json={"status":"success","message":"","timestamp":current_time,"result":{"detections":[]}}
        
        # with open("sample.json", "w") as outfile:
        #     json.dump(output_json, outfile)

    except Exception as e:
        output_json={"status":"failure","message":str(e),"timestamp":current_time}

    print(output_json.keys())
    output_json_str = json.dumps(output_json)

    #convert to bytes
    output_json_bytes = output_json_str.encode()

    # with open('sample.json', 'r') as openfile:
    #    output_json = json.load(openfile)
    # output_json_bytes = json.dumps(output_json).encode()
    return Response(output_json_bytes, media_type="application/json")

if __name__ == '__main__':
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8000
    uvicorn.run(app,host = '0.0.0.0',port=PORT)