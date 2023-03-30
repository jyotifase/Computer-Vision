# https://pyimagesearch.com/2015/05/11/creating-a-face-detection-api-with-python-and-opencv-in-just-5-minutes/
import time, os
import cv2
import numpy as np
import dlib
import imutils
from mtcnn.mtcnn import MTCNN
#from mtcnn import MTCNN
import face_detection
from retinaface import RetinaFace
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def cascade_classifiers_face_detector(image_path):
    # Initialize the cascade classifiers for face
    face_cascade = cv2.CascadeClassifier('G:/RDB_3642/Projects/face_detection/fd_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    # Read image
    img = cv2.imread(image_path)
    # cv2_imshow(img)

    # # Converting image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting the detections
    faces_rect = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5)
    print("faces bbox with given setting:\n",faces_rect)

    # Iterating through rectangles of detected faces
    faces_rect = faces_rect.tolist()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img, faces_rect

def dlib_hog_face_detector(image_path):
    hogFaceDetector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    #image = imutils.resize(image, width=600)
    start = time.time()
    faceRects = hogFaceDetector(img, 0); print(faceRects)
    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))
    faceRects_list = []
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for facerect in faceRects:
        faceRects_list.append([int(facerect.left()), int(facerect.top()), 
                               int(facerect.right()-facerect.left()), int(facerect.bottom()-facerect.top())])
        cv2.rectangle(img, (facerect.left(), facerect.top()), (facerect.right(), facerect.bottom()), (0, 255, 0), 2)
    
    #cv2_imshow(img)
    return img, faceRects_list

def dlib_cnn_face_detector(image_path):
    cnn_face_detect = dlib.cnn_face_detection_model_v1(os.path.join(os.getcwd(),"data","mmod_human_face_detector.dat"))
    img = cv2.imread(image_path)
    #img = cv2.resize(img, (600, 400))
    faces_rect = cnn_face_detect(img, 1); print(faces_rect)
    faces_rect_list = []
    for face in faces_rect:
        x1 = face.rect.left()
        y1 = face.rect.bottom()
        x2 = face.rect.right()
        y2 = face.rect.top()
        w = x2-x1
        h = y1-y2
        faces_rect_list.append([int(x1), int(y1), int(w), int(h)])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return img, faces_rect_list

# def ssd_face_detector(image_path):
#     img = cv2.imread(image_path)
#     output_image = img.copy()
#     # detector = cv2.dnn.readNetFromCaffe(caffeModel="res10_300x300_ssd_iter_140000_fp16.caffemodel", prototxt="deploy.prototxt")    
#     # faces_rect = detector.detect(img)

#     opencv_dnn_model = cv2.dnn.readNetFromCaffe(caffeModel="data/res10_300x300_ssd_iter_140000_fp16.caffemodel", prototxt="data/deploy.prototxt")
#     preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
#                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
#     print(1)
#     opencv_dnn_model.setInput(preprocessed_image)
#     print(2)
#     faces_rect = opencv_dnn_model.forward()
#     print(faces_rect)
#     faces_rect = faces_rect.tolist()
#     for (x, y, w, h) in faces_rect[0,0]:
#         cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return output_image, faces_rect

def ssd_face_detector(image_path):
    img = cv2.imread(image_path)
    output_image = img.copy()

    opencv_dnn_model = cv2.dnn.readNetFromCaffe(caffeModel="data/res10_300x300_ssd_iter_140000_fp16.caffemodel", prototxt="data/deploy.prototxt")
    print(opencv_dnn_model)
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    opencv_dnn_model.setInput(preprocessed_image)
    faces_rect = opencv_dnn_model.forward()
    
    # Reshape output
    faces_rect = faces_rect.reshape(faces_rect.shape[2], faces_rect.shape[3])

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return output_image, faces_rect.tolist()


def mtcnn_face_detector(image_path):
    # Initialize detector
    detector = MTCNN()
    img = cv2.imread(image_path)
    faces_rect = detector.detect_faces(img)
    img_with_dets = img.copy()
    min_size = 50 # minimum width or height of bounding box
    for face in faces_rect:
        box = face['box']
        if box[2] >= min_size and box[3] >= min_size:
            x, y, w, h = box
            keypoints = face['keypoints']
            cv2.rectangle(img_with_dets, (x,y), (x+w,y+h), (0,155,255), 2)
            cv2.circle(img_with_dets, (keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(img_with_dets, (keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(img_with_dets, (keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(img_with_dets, (keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(img_with_dets, (keypoints['mouth_right']), 2, (0,155,255), 2)
    return img_with_dets, faces_rect

def face_detection_detector(image_path):
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
    # Read image
    img = cv2.imread(image_path)
    # Getting detections
    detections = detector.detect(img); print(detections)
    detections = detections.tolist()
    for (x1,y1,x2,y2, c) in detections:
        cv2.rectangle(img, (round(x1),round(y1)),(round(x2),round(y2)), (0, 255, 0), 2)
    return img, detections

def retina_face_detector(image_path):
    # Initialize detector
    img = cv2.imread(image_path)
    detections = RetinaFace.detect_faces(image_path)
    print(detections)
    detections_list =[]
    min_conf = 0.9
    for key in detections:
        if detections[key]["score"] >= min_conf:
            x1,y1,x2,y2 = detections[key]["facial_area"]
            cv2.rectangle(img, (int(x1),int(y1)),(int(x2),int(y2)), (0, 255, 0), 2)
            detections_list.append([int(x1),int(y1),int(x2),int(y2)])
    return img, detections_list

def mediapipe_face_detector(image_path):
    mp_face_detection = mp.solutions.face_detection
    facedetection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)
    img = cv2.imread(image_path)
    results = facedetection.process(img)
    annotated_image = img.copy()
    height, width, _ = img.shape
    detection_results = []
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        xmin = int(bbox.xmin * width)
        ymin = int(bbox.ymin * height)
        xmax = int((bbox.xmin + bbox.width) * width)
        ymax = int((bbox.ymin + bbox.height) * height)
        bbox_points = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        detection_results.append(bbox_points)
        cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #cv2.imwrite('/tmp/annotated_image' + '.png', annotated_image)
    return annotated_image, detection_results

def yunet_face_detector(image_path):
    detector = cv2.FaceDetectorYN.create(os.path.join(os.getcwd(),"data","face_detection_yunet_2022mar.onnx"), "", (320, 320))
    img = cv2.imread(image_path)
    detector.setInputSize((int(img.shape[1]), int(img.shape[0]))) # Set input size
    faces = detector.detect(img)
    print(faces)
    detection_list = []
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
            face_cord = coords[0:4]
            detection_list.append(face_cord.astype(int).tolist())
    return img, detection_list

def facenet_pytorch_mtcnn_face_detector(image_path):
    from facenet_pytorch import MTCNN
    import torch
    import cv2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create face detector
    mtcnn = MTCNN(keep_all=True, select_largest=False, post_process=False, device=device)
    img = cv2.imread(image_path)
    #img = cv2.resize(img, (600, 400))
    bboxes, conf, landmarks= mtcnn.detect(img, landmarks=True)
    bboxes = bboxes.tolist()
    # If there is no confidence that in the frame is a face, don't draw a rectangle around it
    if conf[0] !=  None:
        for (x, y, w, h) in bboxes:
            text = f"{conf[0]*100:.2f}%"
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.putText(img, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(170, 170, 170), 1)
            cv2.rectangle(img, (x, y), (w, h), (255, 255, 255), 1)
    return img, bboxes

image_dir = os.path.join(os.getcwd(),'static')
input_image_path = os.path.join(image_dir,'input_image.png')
