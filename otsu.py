import cv2
import numpy as np
import matplotlib.pyplot as plt


def rgbToGray(image):
    # 𝐼 = 𝑅ound(0.299𝑅 + 0.587𝐺 + 0.114𝐵)
    grayValue = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


def getHistogramArray(grayscaleImage):
    histgoramArray = [0] * 256
    for row in grayscaleImage:
        for pixel in row:
            histgoramArray[pixel] += 1
    return histgoramArray


def showHistogram(grayscaleImage):
    flatList = [item for sublist in grayscaleImage for item in sublist]
    bins = [0] * 256
    for i in range(1, 256):
        bins[i] = bins[i-1] + 1
    plt.hist(flatList, bins=bins)
    plt.show()


def showImage(image, name = "image"):
    # press 0 key to close images
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    originalImage = cv2.imread(r'basket_balls.bmp', 1)
    grayscaleImage = rgbToGray(originalImage)
    showImage(grayscaleImage)
    showHistogram(grayscaleImage)


main()