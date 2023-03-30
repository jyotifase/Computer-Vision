import requests, base64, json, os
API_URL = "http://6392-34-91-199-160.ngrok.io"
API_ENDPOINT = f"{API_URL}/face_detection" #'http://localhost:8000/face_detection'

image_path = 'input_image2.jpg'
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

#algos = ['cascade_classifiers_face_detector', 'dlib_hog_face_detector', 'dlib_cnn_face_detector', 'ssd_face_detector',
#         'mtcnn_face_detector', 'face_detection_detector', 'retina_face_detector', 'mediapipe_face_detector', 'yunet_face_detector',
#         'facenet_pytorch_mtcnn_face_detector']

algos = ['ssd_face_detector','yunet_face_detector']

for i, algo in enumerate(algos):
    json = {"Algorithm":algo,"image":encoded_string}
    try:
        print("Trying to send API request...")
        response = requests.post(API_ENDPOINT, json=json)
        try:
            res_json = response.json()
            if res_json["status"]=="success":
                # if res_json["result"]["classes"] ==[]:
                #     print("Object not present or Not able to detect it.")
                if "image" in res_json["result"]:
                    # Extract the base64 encoded image string from the JSON response
                    image_base64 = res_json["result"]["image"]
                    # Decode the base64 encoded image string
                    image_binary = base64.b64decode(image_base64)
                    # Write the binary data to a file
                    image_name = os.path.join(os.getcwd(),f"output_{i+1}_{algo}.png")
                    with open(image_name, "wb") as f:
                        f.write(image_binary)
                    print(f"{algo}: API response image saved as output_{i}.png")
                else:
                    print(res_json["result"])
            else:
                print("API request sent but, unable to process reuest due to", res_json["message"])
        except Exception as e:
            print("Failed to parse JSON response:", e)
    except Exception as e:
        print("Request failed:", e)
