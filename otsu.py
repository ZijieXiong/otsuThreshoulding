import cv2
import numpy as np

def grayConversion(image):
    grayValue = 0.114 * image[:,:,2] + 0.587 * image[:,:,1] + 0.299 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def histogram(grayScale):
    return

orig = cv2.imread(r'basket_balls.bmp', 1)
g = grayConversion(orig)
cv2.imshow("Original", orig)
cv2.imshow("GrayScale", g)
cv2.waitKey(0)
cv2.destroyAllWindows()
