import cv2
pip install opencv-python
pip install opencv-contrib-python
pip install deepface
from deepface import DeepFace
pip install matplotlib
import matplotlib.pyplot as plt
img=cv2.imread('/content/happy.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
predictions=DeepFace.analyze(img)
predictions
     [{'emotion': {'angry': 2.1513942227666718e-08,
     'disgust': 3.137323053513019e-13,
     'fear': 8.087476745648244e-06,
     'happy': 99.99601841020365,
     'sad': 0.0023413521313289435,
     'surprise': 4.166565546602142e-05,
     'neutral': 0.0015896621258796638},
     'dominant_emotion': 'happy',
     'region': {'x': 252, 'y': 77, 'w': 152, 'h': 152},
     'age': 30,
     'gender': {'Woman': 99.98730421066284, 'Man': 0.012689399591181427},
     'dominant_gender': 'Woman',
     'race': {'asian': 0.010483378719072789,
     'indian': 0.12475807452574372,
     'black': 0.0003589504331102944,
     'white': 70.66368460655212,
     'middle eastern': 21.01241499185562,
     'latino hispanic': 8.188307285308838},
     'dominant_race': 'white'}]

type(predictions)
predictions['dominant_emotion']
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in faces:
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),10)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
            predictions['dominant_emotion'],
            (50,100),
            font,2,
            (0,0,255),
            3,
            cv2.LINE_4);

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
