import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
#用来正常显示中文
plt.rcParams["font.sans-serif"]=["SimHei"]
 

if __name__ == "__main__":
    img = cv2.imread("img/cat1.jpg")
    #img = cv2.resize(img1,300)
    #将图片进行随机裁剪为280×280
    sess = tf.InteractiveSession()
    #显示图片
    # cv2.imwrite("img/crop.jpg",crop_img.eval())

    ######################随机裁剪##################################
    #随机参见
    crop_img = tf.random_crop(img,[280,280,3])
    #通道转换
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    crop_img = cv2.cvtColor(crop_img.eval(),cv2.COLOR_BGR2RGB)   

    ####################随机翻转####################################
    #将图片随机进行水平翻转
    h_flip_img = tf.image.random_flip_left_right(img)      
    #将图片随机进行垂直翻转
    v_flip_img = tf.image.random_flip_up_down(img)
    #通道转换
    h_flip_img = cv2.cvtColor(h_flip_img.eval(),cv2.COLOR_BGR2RGB)
    v_flip_img = cv2.cvtColor(v_flip_img.eval(),cv2.COLOR_BGR2RGB)

    ##########随机亮度、对比度、色度、饱和度的设置###################
    #随机设置图片的亮度
    random_brightness = tf.image.random_brightness(img,max_delta=1)
    random_brightness = cv2.cvtColor(random_brightness.eval(),cv2.COLOR_BGR2RGB)
    #随机设置图片的对比度
    random_contrast = tf.image.random_contrast(img,lower=0.2,upper=1.8)
    random_contrast = cv2.cvtColor(random_contrast.eval(),cv2.COLOR_BGR2RGB)
    #随机设置图片的色度
    random_hue = tf.image.random_hue(img,max_delta=0.3)
    random_hue = cv2.cvtColor(random_hue.eval(),cv2.COLOR_BGR2RGB)

    #随机设置图片的饱和度
    random_satu = tf.image.random_saturation(img,lower=0.2,upper=1.8)
    random_satu = cv2.cvtColor(random_satu.eval(),cv2.COLOR_BGR2RGB)



    plt.figure(1)
    plt.subplot(331)
    plt.imshow(img1)
    plt.title("原始图片")
    plt.subplot(332)
    plt.title("裁剪后的图片")
    plt.imshow(crop_img)

    plt.subplot(334)
    plt.title("水平翻转")
    plt.imshow(h_flip_img)
    plt.subplot(335)
    plt.title("垂直翻转")
    plt.imshow(v_flip_img)

    plt.subplot(336)
    plt.title("亮度")
    plt.imshow(random_brightness)
    plt.subplot(337)
    plt.title("对比度")
    plt.imshow(random_contrast)
    plt.subplot(338)
    plt.title("色度")
    plt.imshow(random_hue)
    plt.subplot(339)
    plt.title("饱和度")
    plt.imshow(random_satu)

    plt.show()
    sess.close()


