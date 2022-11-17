import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from PIL import Image


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
        # normalized original histogram
        # probability of pixel being that region
        weight += histogram[g] / totalArea
    return weight


def otsu_method_two_regions(grayscaleImage):
    grayScaleHistogram = get_grayscale_histogram_array(grayscaleImage)

    smallestVariance = float("inf")
    threshold = 0
    totalArea = len(grayscaleImage)*len(grayscaleImage[0])

    for t in range(256):
        weightB = get_weight_of_area(0, t+1, grayScaleHistogram, totalArea)
        weightF = get_weight_of_area(t+1, 256, grayScaleHistogram, totalArea)

        varianceB = get_variance(grayScaleHistogram[:t+1])
        varianceF = get_variance(grayScaleHistogram[t+1:])

        variance = weightB * varianceB + weightF * varianceF
        if variance < smallestVariance:
            smallestVariance = variance
            threshold = t

    return variance, threshold


def otsu_method_three_regions(grayscaleImage):
    grayScaleHistogram = get_grayscale_histogram_array(grayscaleImage)

    smallestVariance = float("inf")
    threshold1 = 0
    threshold2 = 1
    totalArea = len(grayscaleImage)*len(grayscaleImage[0])
    for t1 in range(256-1):
        weightA = get_weight_of_area(0, t1+1, grayScaleHistogram, totalArea)
        varianceA = get_variance(grayScaleHistogram[:t1+1])
        for t2 in range(t1+1, 256):
            weightB = get_weight_of_area(t1+1, t2+1, grayScaleHistogram, totalArea)
            weightC = get_weight_of_area(t2+1, 256, grayScaleHistogram, totalArea)

            varianceB = get_variance(grayScaleHistogram[t1+1:t2+1])
            varianceC = get_variance(grayScaleHistogram[t2+1:])

            variance = weightA * varianceA + weightB * varianceB + weightC * varianceC
            if variance < smallestVariance:
                smallestVariance = variance
                threshold1 = t1
                threshold2 = t2

    return variance, threshold1, threshold2


def otsu_method_four_regions(grayscaleImage):
    grayScaleHistogram = get_grayscale_histogram_array(grayscaleImage)

    smallestVariance = float("inf")
    threshold1 = 0
    threshold2 = 1
    threshold3 = 2
    totalArea = len(grayscaleImage)*len(grayscaleImage[0])
    for t1 in range(256-2):
        weightA = get_weight_of_area(0, t1+1, grayScaleHistogram, totalArea)
        varianceA = get_variance(grayScaleHistogram[:t1+1])
        for t2 in range(t1+1, 256-1):
            weightB = get_weight_of_area(t1+1, t2+1, grayScaleHistogram, totalArea)
            varianceB = get_variance(grayScaleHistogram[t1+1:t2+1])
            for t3 in range(t2+1, 256):
                weightC = get_weight_of_area(t2+1, t3+1, grayScaleHistogram, totalArea)
                weightD = get_weight_of_area(t3+1, 256, grayScaleHistogram, totalArea)

                varianceC = get_variance(grayScaleHistogram[t2+1:t3+1])
                varianceD = get_variance(grayScaleHistogram[t3+1:])

                variance = weightA * varianceA + weightB * varianceB + weightC * varianceC + weightD * varianceD

                if variance < smallestVariance:
                    smallestVariance = variance
                    threshold1 = t1
                    threshold2 = t2
                    threshold3 = t3

    return variance, threshold1, threshold2, threshold3


#----- The following functions are for image display --------


def show_image(image, name = "image"):
    # press 0 key to close images
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray_to_bicolor(grayscaleImage, threshold):
    biColorImage = copy.deepcopy(grayscaleImage)
    for row in range(len(biColorImage)):
        for col in range(len(biColorImage[row])):
            if biColorImage[row][col] <= threshold:
                biColorImage[row][col] = 0
            else:
                biColorImage[row][col] = 255
    return biColorImage


def gray_to_tricolor(grayscaleImage, threshold1, threshold2):
    triColorImage = copy.deepcopy(grayscaleImage)
    for row in range(len(triColorImage)):
        for col in range(len(triColorImage[row])):
            if triColorImage[row][col] <= threshold1:
                triColorImage[row][col] = 0
            elif triColorImage[row][col] <= threshold2:
                triColorImage[row][col] = 127
            else:
                triColorImage[row][col] = 255
    return triColorImage


def gray_to_quartcolor(grayscaleImage, threshold1, threshold2, threshold3):
    quartColorImage = copy.deepcopy(grayscaleImage)
    for row in range(len(quartColorImage)):
        for col in range(len(quartColorImage[row])):
            if quartColorImage[row][col] <= threshold1:
                quartColorImage[row][col] = 0
            elif quartColorImage[row][col] <= threshold2:
                quartColorImage[row][col] = 85
            elif quartColorImage[row][col] <= threshold3:
                quartColorImage[row][col] = 170
            else:
                quartColorImage[row][col] = 255
    return quartColorImage


#----- The following functions are for testing --------


def output_image(fileName):
    #takes in bmp file only
    image = cv2.imread(fileName+".bmp", 1)
    grayscaleImage = rgb_to_grayscale(image)
    f=open(fileName+'.txt','w')
    f.write("image name: "+fileName+"\t")

    # apply otsu operation with two regions
    varianceTwoRegions, thresholdTwoRegions = otsu_method_two_regions(grayscaleImage)
    f.write("two region: "+str(varianceTwoRegions)+"\t")
    biColorImage = gray_to_bicolor(grayscaleImage, thresholdTwoRegions)
    imageTwoRegion = Image.fromarray(biColorImage)
    imageTwoRegion.save(fileName+"2R.bmp", "bmp")

    # apply otsu operation with three regions
    varianceThreeRegions, thresholdThreeRegions1, thresholdThreeRegions2 = otsu_method_three_regions(grayscaleImage)
    triColorImage = gray_to_tricolor(grayscaleImage, thresholdThreeRegions1, thresholdThreeRegions2)
    f.write("three region: "+str(varianceThreeRegions)+"\t")
    imageThreeRegion = Image.fromarray(triColorImage)
    imageThreeRegion.save(fileName+"3R.bmp", "bmp")

    # apply otsu operation with four regions
    varianceFourRegions, thresholdFourRegions1, thresholdFourRegions2, thresholdFourRegions3 = otsu_method_four_regions(grayscaleImage)
    quartColorImage = gray_to_quartcolor(grayscaleImage, thresholdFourRegions1, thresholdFourRegions2, thresholdFourRegions3)
    f.write("four region: "+str(varianceFourRegions)+"\t")
    imageQuartRegion = Image.fromarray(quartColorImage)
    imageQuartRegion.save(fileName+"4R.bmp", "bmp")

    f.close()


def main():
    output_image('rock-stream1')
    output_image('data13')
    output_image('tiger1')
    output_image('basket_balls')
    output_image('blackroll-duoball')

main()