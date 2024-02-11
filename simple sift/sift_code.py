import cv2
import numpy as np

def disp_img(img, name="image"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_sift(img):

    if img.shape[2] > 1:
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # the following method can be made on 2 steps
    # with 2 method calls, check the documentation
    kp, desc = sift.detectAndCompute(gray,None)

    img = cv2.drawKeypoints(gray,kp,None)
    disp_img(img)

    cv2.drawKeypoints(gray,kp,img,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    disp_img(img)

    return img, desc


img = cv2.imread('home.jpg')
# img = cv2.imread('home-op.jpg')

get_sift(img)