import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np


def CalcOfDamageAndNonDamage(image_name):
    image = cv.imread(image_name)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) #эллипсами покравет всю картинку, цифры-диаметры
    image_erode = cv.erode(image, kernel) #kernel накладывается на картинку и сглаживается
    see_img(image_erode)
    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV) #переводит изображение из одного цветового пространства в другое
    # преобразоввывает RGB/BGR в HSV
    see_img(hsv_img)
    markers = np.zeros((image.shape[0], image.shape[1]), dtype="int32") #делаем массив по np (shape- размер изображения)
    markers[90: 140, 90: 140] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:225] = 1
    markers[236:255, 236:255] = 1
    #Нужно изменить маркеры, что бы они не брали тень.
    leafs_area_BGR = cv.watershed(image_erode, markers) #отделение листа от фона
    healthy_part = cv.inRange(hsv_img, (36, 25, 25), (86, 255, 255))
    see_img(healthy_part)
    ill_part = leafs_area_BGR - healthy_part
    mask = np.zeros_like(image, np.uint8)
    mask[leafs_area_BGR > 1] = (255, 0, 255)
    mask[ill_part > 1] = (0, 0, 255)
    return mask

def read_img(image_name):
    img = cv.imread(image_name)
    if img is None:
        sys.exit("Could not read the image.")
    return img


def see_img(img):
    cv.imshow(" Display window", img)
    k = cv.waitKey(0)
    if k == ord("s"):
        cv.imwrite(" starry_night.png", img)


def CalcOfDamageAndNonDamage_bilateral(image_name):
    image = cv.imread(image_name)

    plt.subplot(321), plt.imshow(image)

    # bilateral = cv.bilateralFilter(image, 30, 75, 75)
    bilateral = cv.bilateralFilter(image, 10, 50, 50)

    plt.subplot(322), plt.imshow(bilateral)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    image_erode = cv.erode(image, kernel) #убирается блеск

    plt.subplot(323), plt.imshow(image_erode)

    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)

    plt.subplot(324), plt.imshow(hsv_img)

    markers = np.zeros((image.shape[0], image.shape[1]), dtype="int32")
    markers[90:140, 90:140] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:225] = 1
    markers[236:255, 236:255] = 1
    leafs_area_BGR = cv.watershed(image_erode, markers)
    healthy_part = cv.inRange(hsv_img, (36, 25, 25), (86, 255, 255))
    ill_part = leafs_area_BGR - healthy_part
    mask = np.zeros_like(image, np.uint8)
    mask[leafs_area_BGR > 1] = (255, 0, 255)
    mask[ill_part > 1] = (0, 0, 255)
    plt.subplot(325), plt.imshow(mask)
    plt.show()
    return mask

def CalcOfDamageAndNonDamage_NonLocalMeans(image_name):
    image = cv.imread(image_name)
    plt.subplot(331), plt.imshow(image)
    b, g, r = cv.split(image)
    rgb_img = cv.merge([r, g, b])
    dst = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    b, g, r = cv.split(dst)
    rgb_dst = cv.merge([r, g, b])
    plt.subplot(332), plt.imshow(rgb_img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) #эллипсами покравет всю картинку, цифры-диаметры
    image_erode = cv.erode(image, kernel) #kernel накладывается на картинку и сглаживается

    plt.subplot(333), plt.imshow(image_erode)

    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)#переводит изображение из одного цветового пространства в другое
    # преобразоввывает RGB/BGR в HSV

    plt.subplot(334), plt.imshow(hsv_img)

    markers = np.zeros((image.shape[0], image.shape[1]), dtype="int32")
    markers[90: 140, 90: 125] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:225] = 1
    markers[236:255, 50:255] = 1
    leafs_area_BGR = cv.watershed(image_erode, markers)
    healthy_part = cv.inRange(hsv_img, (36, 25, 25), (86, 255, 255))
    plt.subplot(335), plt.imshow(healthy_part) #все идеально
    ill_part = leafs_area_BGR - healthy_part
    plt.subplot(336), plt.imshow(ill_part)
    mask = np.zeros_like(image, np.uint8)
    mask[leafs_area_BGR > 1] = (255, 0, 255)
    mask[ill_part > 1] = (0, 0, 255)
    plt.subplot(337), plt.imshow(mask)

    plt.show()
    return mask


sj = '.jpg'
i = 1
while i < 10:
    name = str(i) + sj;
    #CalcOfDamageAndNonDamage_bilateral(name)
    CalcOfDamageAndNonDamage_NonLocalMeans(name)
    i=i+1