import cv2
import pickle
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# load the encoding file
try:
    file = open('EncodeFile.p', 'rb')
    encodeListKnownWithId = pickle.load(file)
    encodeListKnown, peopleId = encodeListKnownWithId
except Exception as e:
    print(f"Error loading encodings: {e}")
finally:
    file.close()


while True:
    success, img = cap.read()

    # resize the image to 1/4 size for faster processing
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgSmall)
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = None
        if len(faceDis) > 0:
            matchIndex = faceDis.argmin()

        if matchIndex is not None and matches[matchIndex]:
            id = peopleId[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, id, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    if not success:
        break

    cv2.imshow("Camara", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break