import numpy as np
import cv2
import os
import time

def filtering_to_library(src, divide_number, mode='average'):
    """
    TODO: opencv 라이브러리를 활용하여 filtering을 진행
    :param src: original image
    :return: filtering 된 image 반환
    """
    if mode == 'average':
        kernel = np.ones((3, 3), np.float32)/divide_number
    elif mode == 'sharpening':
        kernel = np.ones((3, 3), np.float32)/9
        mask = np.array([[0, 0, 0],
                         [0, 2, 0],
                         [0, 0, 0]])
        kernel = mask - kernel

    dst = cv2.filter2D(src, -1, kernel)
    return dst

def my_padding(src, pad_shape, pad_type = 'zero'):
    """
    TODO : padding 구현
    :param src: original image
    :param pad_shape: padding size 2D
    :param pad_type: 'zero' or 'repetition'
    :return:
    """
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w), dtype='uint8')
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        #down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1:]
        #left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        #right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1:p_w + w]
    else:
        print('zero padding')

    return pad_img

def practice_image_filtering(src, kernel):
    """
    TODO : 라이브러리를 사용하지 않고 filtering을 구현
    :param src: original image
    :param kernel: filtering mask
    :return: filtering image
    """

    (h, w) = src.shape
    (k_h, k_w) = kernel.shape
    pad_img = my_padding(src, (k_h, k_w)) #The parameters here are different from before
    dst = np.zeros((h, w)) #output

    # First Four iteration code
    # Second using Numpy slicing
    for i in range(h):
        for j in range(w):
            filtered = min(255, np.sum(pad_img[i:i + k_h, j:j + k_w] * kernel))
            filtered = max(0, filtered)
            dst[i, j] = filtered
    dst = (dst + 0.5).astype(np.uint8)

    return dst

def gaussian_2D_filtering(filter_size, sigma):
    """
    TODO : Generate 2D Gaussian Mask and filter image by generated Gaussain mask
    :param src: original image
    :param filter_size: filter size
    :param sigma: hyperparameter
    :return: Gaussian 2D Filter Mask
    """
    y, x = np.mgrid[-(filter_size // 2):(filter_size // 2) + 1, -(filter_size // 2):(filter_size // 2) + 1]

    return None

def gaussian_1D_filtering(filter_size, sigma):
    """
        TODO : Generate 1D Gaussian Mask and filter image by generated Gaussain mask
        :param src: original image
        :param filter_size: filter size
        :param sigma: hyperparameter
        :return: Gaussian 1D Filter Mask
    """
    x = np.full((1, filter_size), [range(-(filter_size // 2), (filter_size // 2) + 1)])
    return None

def show_image(src, image_name):
    cv2.imshow(image_name, src)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    src_path = os.getcwd() + '/Lena.png'
    src_image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    # 이미지가 잘 불러져 왔는지 체크 안불러져왔으면 Shape is None
    #print(src_image.shape)

    # average_filtering
    library_filtering_image = filtering_to_library(src_image, 9, 'average')
    show_image(library_filtering_image, 'library_image_filtering')
    darken_library_filtering_image = filtering_to_library(src_image, 12, 'average')
    show_image(darken_library_filtering_image, 'darken_library_image_filtering')
    lighten_library_filtering_image = filtering_to_library(src_image, 4, 'average')
    show_image(lighten_library_filtering_image, 'lighten_library_image_filtering')

    # #sharpening_filtering
    library_filtering_image = filtering_to_library(src_image, 9, 'sharpening')
    show_image(library_filtering_image, 'sharpening_library_image_filtering')
    #
    #padding
    library_padding_image = my_padding(src_image, (9, 9), 'zero')
    show_image(library_padding_image, 'zero_padding_image')
    library_padding_image = my_padding(src_image, (9, 9), 'repetition')
    show_image(library_padding_image, 'repetition_padding_image')

    # ----------- Practice -----------
    first_kernel = np.ones((3,3)) / 9.

    library_padding_image = practice_image_filtering(src_image, first_kernel)
    show_image(library_padding_image, 'repetition_padding_image')

    # Gaussain 2D Mask
    # time check
    start = time.perf_counter()
    gaus2D = gaussian_2D_filtering(5, 1)
    library_padding_image = practice_image_filtering(src_image, gaus2D)
    print('2D Gaussian Filter Time : {}'.format(time.perf_counter() - start))
    show_image(library_padding_image, 'Gaussian_2D_filtering_image')

    # Gaussian 1D Mask
    start = time.perf_counter()
    gaus1D = gaussian_1D_filtering(5, 1)
    dst_gaus1D = practice_image_filtering(src_image, gaus1D)
    dst_gaus1D = practice_image_filtering(dst_gaus1D, gaus1D.T)
    dst_gaus1D = (dst_gaus1D + 0.5).astype(np.uint8)
    print('1D Gaussian Filter Time : {}'.format(time.perf_counter() - start))
    show_image(dst_gaus1D, 'Gaussian_1D_filtering_image')
