import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read images
iname1 = "img/1.png"
iname2 = "img/2.jpg"
iname3 = "img/3.jpg"
iname5 = "img/5.jpg"

image1 = cv2.imread(iname1, 0)
image2 = cv2.imread(iname2)
image3 = cv2.imread(iname3)
image5 = cv2.imread(iname5) 

def amBan(image):
    return 255 - image

def chuyen_doi_log(image, c):
    return c * (np.log(1 + image))

def chuyen_doi_mu(image, g, c):
    return c * np.pow(image, g)

def demo_log():
    image1 = cv2.imread(iname1, 0)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image1, cmap="gray")
    plt.title("Ảnh gốc")
    plt.axis("off")
    c = 255 / np.log(1 + np.max(image1))
    image1 = chuyen_doi_log(image1, c)
    image1 = np.array(image1, dtype=np.uint8)
    plt.subplot(1,2,2)
    plt.imshow(image1, cmap="gray")
    plt.title("Ảnh sau log transform")
    plt.axis("off")

    plt.show()
    

def demo_mu():
    image1 = cv2.imread(iname1, 0)
    g = 1
    c = 1.0 / np.pow(np.max(image1), g)

    image1 = cv2.imread(iname1, 0)
    image1 = chuyen_doi_mu(image1, g, c)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(image1, cmap="gray")
    plt.title("Ảnh dùng c chuẩn")
    plt.axis("off")
    print(image1)

    c = 2
    image1 = cv2.imread(iname1, 0)
    image1 = chuyen_doi_mu(image1, g, c)
    plt.subplot(1,2,2)
    plt.imshow(image1, cmap="gray")
    plt.title("Ảnh sau chuyển đổi mũ c=2")
    plt.axis("off")

    plt.show()
    
def cal_img(npArray):
    arr = np.bincount(npArray.flatten(), minlength=256)
    pdf = arr / npArray.size
    cdf = np.cumsum(pdf)
    # cdf_norm = (cdf.max
    tf = np.round(cdf * 255)
    img = tf[npArray]
    return img


def demo_histogram_equalization(image):
    r, g, b = cv2.split(image)
    tf_r = cal_img(r)
    tf_g = cal_img(g)
    tf_b = cal_img(b)

    img_refine = cv2.merge((tf_r, tf_g, tf_b))
    plt.imshow(img_refine.astype(np.uint8))
    
demo_histogram_equalization(cv2.imread("img/blur.png"))