import cv2

img1 = cv2.imread("person.jpg")
cv2.imshow('original image', img1)
cv2.imwrite("saved_original_image.jpg". img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

resized = cv2.resize(img1, (640, 480)) #width,height\
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img1, (5, 5), 0)
cv2.Canny(img1, 100, 200)

import numpy as np 

canvas = np.zeros((512, 512 ))

cap = cv2.VideoCapture(0)

frames = []
gap = 5
count = 0 
MIN_AREA = 1000

while True:
    