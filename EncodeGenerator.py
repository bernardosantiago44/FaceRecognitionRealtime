import cv2
import face_recognition
import pickle
import os

folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []

peopleId = []

for path in pathList:
    #ignore system files like .DS_Store
    if path.startswith('.'):
        continue

    img = cv2.imread(os.path.join(folderPath, path))
    imgList.append(img)

    id = os.path.splitext(path)[0]
    peopleId.append(id)

def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList

print('Encoding Started...')
encodeListKnown = findEncodings(imgList)
encodeListKnownWithId = [encodeListKnown, peopleId]
print('Encoding Complete')

try:
    file = open('EncodeFile.p', 'wb')
    pickle.dump(encodeListKnownWithId, file)
except Exception as e:
    print(f"Error saving encodings: {e}")
finally:
    file.close()