import cv2
import numpy as np
import os


def sobel_convolution(sample_name):
    image = cv2.imread(sample_name)
    dst = image.copy()
    cv2.namedWindow("Sobel", cv2.WINDOW_NORMAL)

    kernel_matrix = np.array([[-1.0, 0.0, 1.0],
                              [-2.0, 0.0, 2.0],
                              [-1.0, 0.0, 1.0]])

    cv2.filter2D(src=image, dst=dst, ddepth=-1, kernel=kernel_matrix)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

    cv2.imshow("Sobel", dst)


def convolution(sample_name):
    image = cv2.imread(sample_name)
    dst = image.copy()

    cv2.namedWindow("LabConvolution", cv2.WINDOW_NORMAL)

    kernel_matrix = np.array([[-1.0, -1.0, -1.0],
                              [-1.0, 8.0, -1.0],
                              [-1.0, -1.0, -1.0]])

    cv2.filter2D(src=image, dst=dst, ddepth=-1, kernel=kernel_matrix)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

    cv2.imshow("LabConvolution", dst)


def contours(sample_name):
    image = cv2.imread(sample_name, cv2.IMREAD_UNCHANGED)
    cv2.namedWindow("contours", cv2.WINDOW_NORMAL)

    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = 160
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    contours_, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = np.zeros(image.shape)
    cv2.drawContours(img_contours, contours_, -1, (255, 255, 255), 1)

    cv2.imshow("contours", img_contours)


def show(sample_name):
    image = cv2.imread('images/' + sample_name, cv2.IMREAD_UNCHANGED)
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.imshow("original", image)

    sobel_convolution('images/' + file)
    convolution('images/' + file)
    contours('images/' + file)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    currentPath = os.getcwd()
    imagesDirectory = os.path.join(currentPath, "images")
    for dirPath, _, folderContent in os.walk(imagesDirectory):
        for file in folderContent:
            show(file)
