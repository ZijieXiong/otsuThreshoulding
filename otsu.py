import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


#----- otsu thresholding operation --------

def rgb_to_grayscale(image):
    # convert an image to grayscale and return the grayscale image

    # 𝐼 = 𝑅ound(0.299𝑅 + 0.587𝐺 + 0.114𝐵)
    grayValue = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    grayImg = grayValue.astype(np.uint8)
    return grayImg


def get_grayscale_histogram_array(grayscaleImage):
    # return an array of grayscale value, where
    #  - index : grayscale value, range from 0 to 255
    #  - histogram[index] : number of pixels belonging to this grayscale value
    histgoramArray = [0] * 256
    for row in grayscaleImage:
        for pixel in row:
            histgoramArray[pixel] += 1
    return histgoramArray


def get_mean(histogram):
    sum = 0
    count = 0
    # g stands for grayscale value
    # histogram[g] is the number of pixels with this grayscale value
    for g in range(len(histogram)):
        sum += g * histogram[g]
        count += histogram[g]

    if count == 0:
        return 0
    return sum / count


def get_variance(histogram):
    mean = get_mean(histogram)
    sumOfSquares = 0
    count = 0
    # g stands for grayscale value
    # histogram[g] is the number of pixels with this grayscale value
    for g in range(len(histogram)):
        sumOfSquares += ((g - mean) * (g - mean)) * histogram[g]
        count += histogram[g]

    if count == 0:
        return 0
    return sumOfSquares / count


def get_area(left, right, histogram):
    # g stands for grayscale value
    # histogram[g] is the number of pixels with this grayscale value
    area = 0
    for g in range(left, right):
        area += histogram[g]
    return area


def get_weight_of_area(left, right, histogram, totalArea):
    # g stands for grayscale value
    # histogram[g] is the number of pixels with this grayscale value
    weight = 0
    for g in range(left, right):
        weight += histogram[g] / totalArea
    return weight


def otsu_method_two_regions(grayscaleImage):
    grayScaleHistogram = get_grayscale_histogram_array(grayscaleImage)

    smallestVariance = float("inf")
    threshold = 0

    for t in range(256):
        backgroundArea = get_area(0, t+1, grayScaleHistogram)
        foregroundArea = get_area(t+1, 256, grayScaleHistogram)
        totalArea = backgroundArea + foregroundArea

        weightB = get_weight_of_area(0, t+1, grayScaleHistogram, totalArea)
        weightF = get_weight_of_area(t+1, 256, grayScaleHistogram, totalArea)

        varianceB = get_variance(grayScaleHistogram[:t+1])
        varianceF = get_variance(grayScaleHistogram[t+1:])

        variance = weightB * varianceB + weightF * varianceF
        if variance < smallestVariance:
            smallestVariance = variance
            threshold = t

    return threshold


#----- The following functions are for image display --------

def showImage(image, name = "image"):
    # press 0 key to close images
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showHistogram(grayscaleImage):
    flatList = [item for sublist in grayscaleImage for item in sublist]
    bins = [0] * 256
    for i in range(1, 256):
        bins[i] = bins[i-1] + 1
    plt.hist(flatList, bins=bins)
    plt.show()


def grayToBicolor(grayscaleImage, threshold):
    bicolorImage = copy.deepcopy(grayscaleImage)
    for row in range(len(bicolorImage)):
        for col in range(len(bicolorImage[row])):
            if bicolorImage[row][col] <= threshold:
                bicolorImage[row][col] = 0
            else:
                bicolorImage[row][col] = 255
    return bicolorImage


#----- The following functions are for testing --------

def testCase1():
    image = cv2.imread(r'data13.bmp', 1)
    grayscaleImage = rgb_to_grayscale(image)
    threshold = otsu_method_two_regions(grayscaleImage)
    biColorImage = grayToBicolor(grayscaleImage, threshold)
    showImage(biColorImage)

def main():
    testCase1()

main()