import cv2
import numpy as np

image_template = cv2.imread('number_finding(1).png')
cv2.imshow("Original Template", image_template)

height, width = image_template.shape[:2]

original_image = cv2.imread('number_finding.png')
cv2.imshow("Original Image", original_image)

gray_template = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
gray_template=cv2.GaussianBlur(gray_template,(7,7),0)
gray_original=cv2.GaussianBlur(gray_original,(13,13),0)
gray_template = cv2.dilate(gray_template, (1, 1), iterations=0)
gray_original = cv2.dilate(gray_original, (1, 1), iterations=0)

match = cv2.matchTemplate(gray_original, gray_template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

top_left = max_loc
bottom_right = (top_left[0]+height, top_left[1]+width)
cv2.rectangle(original_image, top_left, bottom_right, (0,0,255), 5)

cv2.imshow("Original Image with matched area", original_image)
cv2.waitKey(0)

cv2.destroyAllWindows()