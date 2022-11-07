import cv2
import numpy as np
import os


def save_to_show(array, height, width):
    rgb = np.reshape(array, (height * width, 3))
    rgb = np.reshape(rgb, (height, width, 3))
    cv2.imwrite("temp/temp.png", rgb)
    dst = cv2.imread("temp/temp.png")
    cv2.namedWindow("myFunction", cv2.WINDOW_NORMAL)
    cv2.imshow('myFunction', dst)


def check_rgb_range(num):
    if num < 0:
        num = 0
    elif num > 255:
        num = 255

    return num


def really_my(sample_name, kernel_matrix):
    image = cv2.imread(sample_name)

    byte_array = image.reshape(-1)
    byte_result = np.full((len(byte_array)), 0, dtype=int)
    shift, width, _ = image.shape

    for i in range(0, len(byte_array) - 2 * shift, 3):
        r_sum, g_sum, b_sum = 0, 0, 0

        for kr in range(3):
            for kc in range(3):
                r_sum += byte_array[shift + i + kc * 3 + 0] * kernel_matrix[kr][kc]
                g_sum += byte_array[shift + i + kc * 3 + 1] * kernel_matrix[kr][kc]
                b_sum += byte_array[shift + i + kc * 3 + 2] * kernel_matrix[kr][kc]

        r_sum = check_rgb_range(r_sum)
        g_sum = check_rgb_range(g_sum)
        b_sum = check_rgb_range(b_sum)

        byte_result[shift + i + 0] = r_sum
        byte_result[shift + i + 1] = g_sum
        byte_result[shift + i + 2] = b_sum

    save_to_show(byte_result, shift, width)


def sobel_convolution(sample_name):
    image = cv2.imread(sample_name)
    dst = image.copy()
    cv2.namedWindow("Sobel", cv2.WINDOW_NORMAL)

    kernel_matrix = np.array([[-1.0, 0.0, 1.0],
                              [-2.0, 0.0, 2.0],
                              [-1.0, 0.0, 1.0]])

    cv2.filter2D(src=image, dst=dst, ddepth=-1, kernel=kernel_matrix)
    # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

    cv2.imshow("Sobel", dst)


def convolution(sample_name):
    image = cv2.imread(sample_name)
    dst = image.copy()

    cv2.namedWindow("LabConvolution", cv2.WINDOW_NORMAL)

    kernel_matrix = np.array([[-1.0, -1.0, -1.0],
                              [-1.0, 8.0, -1.0],
                              [-1.0, -1.0, -1.0]])

    cv2.filter2D(src=image, dst=dst, ddepth=-1, kernel=kernel_matrix)
    # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

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

    sobel_convolution('images/' + sample_name)
    convolution('images/' + sample_name)
    contours('images/' + sample_name)
    really_my('images/' + sample_name,
              np.array([[-1.0, 0.0, 1.0],
                        [-2.0, 0.0, 2.0],
                        [-1.0, 0.0, 1.0]])
              # np.array([[-1.0, -1.0, -1.0],
              #           [-1.0, 8.0, -1.0],
              #           [-1.0, -1.0, -1.0]])
              # np.array([[0.0, -1.0, 0.0],
              #           [-1.0, 8.0, -1.0],
              #           [0.0, -1.0, 0.0]])
              # np.array([[-1.0, 0.0, -1.0],
              #           [0.0, 8.0, 0.0],
              #           [-1.0, 0.0, -1.0]])
              )

    cv2.waitKey()
    cv2.destroyAllWindows()


def run():
    current_path = os.getcwd()
    images_directory = os.path.join(current_path, "images")
    for dirPath, _, folderContent in os.walk(images_directory):
        for file in folderContent:
            show(file)


if __name__ == '__main__':
    run()
