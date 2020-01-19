import cv2 as cv
import numpy as np

image = np.random.normal(size=(500, 500, 3))

print(image.shape)

cv.imshow("original", image)

image[:10, :100, :] = (0, 255, 0)

cv.imshow("original", image)

cv.rectangle(image, (0, 0), (10, 100), (0, 0, 255))
cv.imshow("rectangle", image)

pascal_voc_image = cv.imread("2007_000027.jpg")

cv.imshow("pascal", pascal_voc_image)

# < width > 486 < / width >
# < height > 500 < / height >
# < depth > 3 < / depth >

# < xmin > 174 < / xmin >
# < ymin > 101 < / ymin >
# < xmax > 349 < / xmax >
# < ymax > 351 < / ymax >

left_up_base = np.array(pascal_voc_image)
left_up_base[174:349, 101:351, :] -= 100
cv.imshow("left up", left_up_base)

left_down_base = np.array(pascal_voc_image)
left_down_base[500 - 351:500 - 101, 174:349, :] -= 100
cv.imshow("left down", left_down_base)

left_down_base2 = np.array(pascal_voc_image)
left_down_base2[101:351, 174:349, :] -= 100
cv.imshow("left down 2", left_down_base2)
cv.waitKey()
