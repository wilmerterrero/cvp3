import cv2
from .recognize import Recognize

cp = cv2.VideoCapture(0,cv2.CAP_DSHOW)
Recognize.RecognizeFace(cp)
    
cp.release()
cv2.destroyAllWindows()