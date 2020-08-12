from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import tensorflow as tf
import cv2
MODEL=tf.keras.models.load_model("model.h5")
classes=['anger','disgust','happy','neutral','sad','surprise']
def load_image(filename):
    img=cv2.imread(filename,0)
    img=cv2.resize(img,(128,128))
    data = np.asarray(img,dtype="int32")
    return data/255.0

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)


data = [load_image(filename)]
data=np.array(data)
data = np.array(data).reshape(-1,128,128,1)



j=MODEL.predict([data])
i=np.argmax(j)

print('Emotion:{0}'.format(classes[i]))

while True:
    img=cv2.imread(filename)
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()


