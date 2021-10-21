import cv2 as cv
import sys
import matplotlib.pyplot as plt
import numpy as np


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


def CalcOfDamageAndNonDamage(image_name):
    image = cv.imread(image_name)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    image_erode = cv.erode(image, kernel) #убирается блеск
    see_img(image_erode)
    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)
    see_img(hsv_img)
    markers = np.zeros((image.shape[0], image.shape[1]), dtype="int32")
    markers[90: 140, 90: 140] = 255
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
    return mask


sj = '.jpg'
i = 2
while i < 4:
    name = str(i) + sj;
    see_img(read_img(name))
    see_img(CalcOfDamageAndNonDamage(name))
    i=i+1
# img = cv.imread("1.jpg")
# (b, g, r) = img[0, 0]
# print("Red : {} , Green : {} , Blue : {}".format(r, g, b))
# if img is None:
#     sys.exit("Could not read the image. ")
# cv.imshow(" Display window", img)
# k = cv.waitKey(0)
# if k == ord("s"):
#     cv.imwrite(" starry_night.png", img)
# CalcOfDamageAndNonDamage("1.jpg")

# img = cv.imread("1.jpg")
# px = img[100, 100]  # get BGR print (px)
# # accessing only blue pixel blue = img[100 ,100 ,0]
# print(blue)
# # modify pixel img[100 ,100] = [255 ,255 ,255]
# # fast	accessing RED value
# print(img.item(10, 10, 2))  # modifying RED value img . itemset ((10 ,10 ,2) ,100)
