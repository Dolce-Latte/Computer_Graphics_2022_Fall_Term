import numpy as np
import cv2
import os
import time

def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return padding_img

def my_filtering(src, filter):
    (h, w) = src.shape
    (m_h, m_w) = filter.shape
    pad_img = my_padding(src, filter)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * filter)
    return dst

def generate_sobel_filter():

    return None

def calculate_magnitude(sobel_x, sobel_y):

    # Element Wise Multiplication



    # Simple

    return None

def normalize(src):
    src = (((src - np.min(src)) / (np.max(src) - np.min(src))) * 255 + 0.5).astype(np.uint8)
    return src

def cv2_sobel(src):

    return None

def threshold(src, min_threshold):
    ret, dst = cv2.threshold(src, min_threshold, 255, cv2.THRESH_BINARY)
    return dst

def gaussian_2D_filtering(filter_size, sigma):

    y, x = np.mgrid[-(filter_size // 2) : (filter_size // 2) + 1, -(filter_size // 2) : (filter_size // 2) + 1]

    scalar = 1 / (2 * np.pi * sigma ** 2)
    exponential = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    mask = scalar * exponential
    gaussian_mask = mask / np.sum(mask)

    return gaussian_mask

def get_DoG_filter(fsize, sigma=1):
    y,x = np.mgrid[-(fsize // 2) : (fsize // 2) + 1, -(fsize // 2) : (fsize // 2) + 1]

    return DoG_x, DoG_y

if __name__ == "__main__":
    src_path = 'Yujin.jpg'
    src_image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

    # src_image = cv2.resize(src_image, (640,480))
    # sobel_x, sobel_y = generate_sobel_filter()
    #
    # sobel_x_yujin = my_filtering(src_image, sobel_x)
    # sobel_y_yujin = my_filtering(src_image, sobel_y)
    # sobel_yujin = calculate_magnitude(sobel_x_yujin, sobel_y_yujin)
    # sobel_x_yujin = normalize(sobel_x_yujin)
    # sobel_y_yujin = normalize(sobel_y_yujin)
    #
    # thrs_image = threshold(sobel_yujin, 80)
    #
    # cv2.imshow('Yujin',src_image)
    # cv2.imshow('Yujin_sobel_x', sobel_x_yujin)
    # cv2.imshow('Yujin_sobel_y', sobel_y_yujin)
    # cv2.imshow('Yujin_sobel', (sobel_yujin + 0.5).astype(np.uint8) )
    # cv2.imshow('Yujin_sobel_threshold', (thrs_image + 0.5).astype(np.uint8))
    # cv2.waitKey()
    # cv2.destoryAllWindows()

    # sobx, soby = cv2_sobel(src_image)
    # cv2.imshow('Yujin',src_image)
    # cv2.imshow('Yujin_sobel_x', sobx / 255)
    # cv2.imshow('Yujin_sobel_y', soby / 255)
    # cv2.waitKey()
    # cv2.destoryAllWindows()

    # threshold
    # threshold_test_image = cv2.imread('threshold_test.png',cv2.IMREAD_GRAYSCALE)
    # thrs_image = threshold(threshold_test_image, 100)
    # cv2.imshow('threshold_image', thrs_image)
    # cv2.waitKey()
    # cv2.destoryAllWindows()

    # Derivative of Gaussian
    # DoG_x, DoG_y = get_DoG_filter(5, 3)
    # dst_x = my_filtering(src_image, DoG_x)
    # dst_y = my_filtering(src_image, DoG_y)

    # gaus = gaussian_2D_filtering(5, 1)
    # sobel_x, sobel_y = generate_sobel_filter()
    # gaus_filtering = my_filtering(src_image, gaus)
    # dst_x = my_filtering(gaus_filtering, sobel_x)
    # dst_y = my_filtering(gaus_filtering, sobel_y)

    # dst = calculate_magnitude(dst_x, dst_y)
    # cv2.imshow('Yujin',src_image)
    # cv2.imshow('Yujin_DoG_x', dst_x / 255)
    # cv2.imshow('Yujin_DoG_y', dst_y / 255)
    # cv2.imshow('Yujin_DoG', dst / 255)
    # cv2.waitKey()
    # cv2.destoryAllWindows()

