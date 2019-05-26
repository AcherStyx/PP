import numpy as np

import cv2

image = cv2.imread("./sample.jpg")


b  = np.array([[[100,100],  [250,100], [300,220],[100,230]]], dtype = np.int32)

im = np.zeros(image.shape[:2], dtype = "uint8")
#cv2.polylines(im, b, 1, 255)
cv2.fillPoly(im, b, 255)

mask = im
cv2.imshow("Mask", mask)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask to Image", masked)
cv2.waitKey(0)