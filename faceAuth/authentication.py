# import cv2
# import numpy as np
# import os
# import face_recognition as fr
# from firebase import Firebase
# from google.cloud import storage
#
# config = {
#     "apiKey": "AIzaSyATJUWiR18Mfc_Yrd4CTAqZVwn-pVpXnno",
#     "authDomain": "signin-example-b3f10.firebaseapp.com",
#     "databaseURL": "https://signin-example-b3f10-default-rtdb.firebaseio.com",
#     "storageBucket": "signin-example-b3f10.appspot.com",
#     "serviceAccount": "serviceAccountKey.json"
# }
#
# firebase = Firebase(config)
# store = firebase.storage()
#
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/ZENBOOK/Raspberry-Pi-Module/faceAuth/serviceAccountKey.json"  # ./faceAuth/serviceAccountKey.json
#
# storage_client = storage.Client()
# bucket = "signin-example-b3f10.appspot.com"
#
# bucket = storage_client.get_bucket(bucket)
# blobs = list(bucket.list_blobs())
# for blob in blobs:
#     store.child(blob.name).download("C:/Users/ZENBOOK/Raspberry-Pi-Module/train/" + blob.name[5:] + ".jpeg")
# #
# # # ----------------------------------------------------------------
# #
# # cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# # cam.set(3, 640)  # set video width
# # cam.set(4, 480)  # set video height
# #
# # face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")    # Loading the required haar-cascade xml classifier file
# # count = 0
# #
# # while (True):
# #
# #     ret, img = cam.read()
# #     # img = cv2.flip(img, -1) # flip video image vertically
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     faces = face_detector.detectMultiScale(gray, 1.3, 5)    # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
# #
# #     for (x, y, w, h) in faces:
# #         count += 1
# #         cv2.imwrite("test/compared.jpg", gray[y:y + h, x:x + w])    # Save the captured image into the datasets folder
# #
# #     if count >= 30:  # Take 30 face sample and stop video
# #         break
# #
# # cam.release()
# # cv2.destroyAllWindows()
# #
# # path = "./train/"
# #
# # known_names = []
# # known_name_encodings = []
# #
# # images = os.listdir(path)
# # for _ in images:
# #     image = fr.load_image_file(path + _)
# #     image_path = path + _
# #     encoding = fr.face_encodings(image)[0]
# #     known_name_encodings.append(encoding)
# #     known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())
# #
# # test_image = "./test/compared.jpg"
# #
# # image = cv2.imread(test_image)
# # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #
# # face_locations = fr.face_locations(image)
# # face_encodings = fr.face_encodings(image, face_locations)
# #
# # for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
# #     matches = fr.compare_faces(known_name_encodings, face_encoding, tolerance=0.5)
# #     name = "Unknown Person"
# #
# #     face_distances = fr.face_distance(known_name_encodings, face_encoding)
# #     best_match = np.argmin(face_distances)
# #
# #     if matches[best_match]:
# #         name = known_names[best_match]
# #
# # print(name)
