# openCV-basic

[TOC]

## 1、openCV的概述

OpenCV是一个基于BSD许可（开源）发行的跨平台计算机视觉库，可以运行在Linux、Windows、Android和Mac OS操作系统上。它轻量级而且高效——由一系列 C 函数和少量 C++ 类构成，同时提供了Python、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法(百度百科)。

## 2、环境依赖和opencv包

- ###  读取图片，将其转换为数组

-  数组数据转换

- 数组数据窗口展示

- 图像保存

- 图像的截取

- BGR数据切片

- 同样大小的数组像素值运算

- 图片的融合

- 图片的比例缩放

## 3、基本使用

- 前提：准备若干张图片到本地

- 　一张图片是由很多个像素点组成，对于计算机而言，最终呈现在用户面前的是由每个像素点的值所决定（0~255），0对应黑色，255对应白色。我们在生活中通常接触的都是彩色图片，由RGB三通道共同构成一张上面的彩色图片，每一个通道对应的像素值反映出其亮度（三个通道可以理解成三个矩阵）。而灰度图像通常只有一个颜色通道来表现

  ```python
  from matplotlib import pyplot as pyl
  import cv2
  import numpy
  
  # 1、读取图片，将其转换为数组
  img = cv2.imread("cat.jpg")
  # img是一个numpy.ndarray对象，默认是以BGR三通道读取图片数据（三维数组）
  # img_gray = cv2.imread("cat.jpg",cv2.IMREAD_GRAYSCALE)   以灰度图像方式读取图片数据（二维数组)
  
  
  # 2、数组数据转换
  img_BGR = cv2.imread("cat.jpg")
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将其转换为灰度的二维数组数据
  
  # 3、数组数据窗口展示
  img = cv2.imread("cat.jpg")
  cv2.imshow("IMage", img)
  cv2.waitKey(0)  # 按任意键关闭窗口，cv2.waitKey(1000) 延时一秒关闭窗口
  cv2.destroyAllWindows()
  
  # 4、图像保存
  cv2.imwrite("mycat.jpg", img)
  
  # 5、图像截取
  # 其实本质就是对np数组进行操作
  img = cv2.imread("cat.jpg")
  cv2.imshow("IMage", img[:100, :200])  # 取前100行，前200列的像素作为图像展示
  
  # 6、BGR数据切片
  img = cv2.imread("cat.jpg")
  # 切片
  b, g, r = cv2.split(img)  # 得到各自颜色通道的二维数组数据
  # 合并
  img = cv2.merge([b, g, r])
  
  # 7、 同样大小的数组像素值运算
  img = cv2.imread("cat.jpg")
  img_2 = numpy.copy(img)
  # np相加,像素值只要超过255，就减掉255，比如257，结果就为2
  print(img[:3, :3, 0] + img_2[:3, :3, 0])
  # cv2相加,像素值超过255，就等于255
  print(cv2.add(img[:3, :3, 0], img_2[:3, :3, 0]))
  
  # 8 图片的融合
  img_cat = cv2.imread("cat.jpg")
  img_dog = cv2.imread("dog.jpg")
  
  ret = cv2.addWeighted(img_cat, 0.2, img_dog, 0.8, 0)  # 数据后面的值决定图片融合和所占的权重
  cv2.imshow("IMage", ret)
  cv2.waitKey(0)  # 按任意键关闭窗口，cv2.waitKey(1000) 延时一秒关闭窗口
  cv2.destroyAllWindows()  # 如果图片大小不一致，使用cv2.resize(img_xx,(300,200))  ————》转换为np.shape = 200,300的数组
  
  # 9 图片的比例缩放
  img_cat = cv2.imread("cat.jpg")
  ret = cv2.resize(img_cat, (0, 0), fx=3, fy=1)  # 横向拉长三倍
  ret2 = cv2.resize(img_cat, (0, 0), fx=3, fy=3)  # 图片扩大三倍
  
  ```

### 函数总结

### 1、imread函数

imread函数读取数字图像，官网对于该函数的定义

```python
cv2.imread(path_of_image, intflag)
```

- 函数参数一： 需要读入图像的完整的路径

- 函数参数二： 标志以什么形式读入图像，可以选择一下方式：

  -  cv2.IMREAD_COLOR： 加载彩色图像。任何图像的透明度都将被忽略。它是默认标志

  -  cv2.IMREAD_GRAYSCALE：以灰度模式加载图像

  -  cv2.IMREAD_UNCHANGED：保留读取图片原有的颜色通道

    -  1 ：等同于cv2.IMREAD_COLOR

    - 0 ：等同于cv2.IMREAD_GRAYSCALE

    - -1 ：等同于cv2.IMREAD_UNCHANGED

      ```python
      import numpy as np
      import cv2
      
      gray_img = cv2.imread('img/cartoon.jpg', 0)  #加载灰度图像
      rgb_img = cv2.imread('img/cartoon.jpg', 1)   #加载RGB彩色图像
      ```

### 2、imshow函数

imshow函数作用是在窗口中显示图像，窗口自动适合于图像大小，也可以通过imutils模块调整显示图像的窗口的大小。函数官方定义如下：

```python
cv2.imshow(windows_name, image)
```

- 函数参数一： 窗口名称(字符串)
- 函数参数二： 图像对象，类型是numpy中的ndarray类型，注：这里可以通过imutils模块改变图像显示大小，下面示例展示

```python
cv2.imshow('origin image', rgb_img)   #显示原图
cv2.imshow('origin image', imutils.resize(rgb_img, 800))  #利用imutils模块调整显示图像大小
cv2.imshow('gray image', imutils.resize(gray_img, 800))
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
```



### 3、imwrite函数

imwrite函数检图像保存到本地，官方定义：

```python
cv2.imwrite(image_filename, image)
```

- 函数参数一： 保存的图像名称(字符串)
- 函数参数二： 图像对象，类型是numpy中的ndarray类型


```python
cv2.imwrite('rgb_img.jpg', rgb_img)   #将图像保存成jpg文件
cv2.imwrite('gray_img.png', gray_img) #将图像保存成png文件
```



### 4、窗口销毁函数

当使用imshow函数展示图像时，**最后需要在程序中对图像展示窗口进行销毁，否则程序将无法正常终止**，常用的销毁窗口的函数有下面两个：

1. cv2.destroyWindow(windows_name) #销毁单个特定窗口
   参数： 将要销毁的窗口的名字
2. cv2.destroyAllWindows() #销毁全部窗口，无参数


那我们合适销毁窗口，肯定不能图片窗口一出现我们就将窗口销毁，这样便没法观看窗口，试想有两种方式：
(1) 让窗口停留一段时间然后自动销毁；
(2) 接收指定的命令，如接收指定的键盘敲击然后结束我们想要结束的窗口
以上两种情况都将使用cv2.waitKey函数， 函数定义：

```python
cv2.waitKey(time_of_milliseconds)
```

唯一参数 time_of_milliseconds是整数，可正可负也可是零，含义和操作也不同，分别对应上面说的两种情况

1. time_of_milliseconds > 0 ：此时time_of_milliseconds表示时间，单位是毫秒，含义表示等待 time_of_milliseconds毫秒后图像将自动销毁

   ```python
   #表示等待10秒后，将销毁所有图像
   if cv2.waitKey(10000):
       cv2.destroyAllWindows()
   
   #表示等待10秒，将销毁窗口名称为'origin image'的图像窗口
   if cv2.waitKey(10000):
       cv2.destroyWindow('origin image')
   ```

2.  time_of_milliseconds <= 0 ： 此时图像窗口将等待一个键盘敲击，接收到指定的键盘敲击便会进行窗口销毁。我们可以自定义等待敲击的键盘，通过下面的例子进行更好的解释

   ```python
   #当指定waitKey(0) == 27时当敲击键盘 Esc 时便销毁所有窗口
   if cv2.waitKey(0) == 27:
       cv2.destroyAllWindows()
   
   #当接收到键盘敲击A时，便销毁名称为'origin image'的图像窗口
   if cv2.waitKey(-1) == ord('A'):
       cv2.destroyWindow('origin image')
   ```

   

## 4、感想

网上教程的例子一定要亲自跑一下，一方面可以让自己印象更加深刻，比如这回一开始自己尝试cv2.imshow（),发现程序始终无法正常终止，之后结合教程才认识到当使用imshow函数展示图像时，最后需要在程序中对图像展示窗口进行销毁，否则程序将无法正常终止，另一方面是网上的例子总是避免不了各式各样的不严谨的错误。



参考博文

[opencv——基础篇](https://www.cnblogs.com/lufengyu/p/11495148.html)

[Python3与OpenCV3.3 图像处理（一）--环境搭建与简单DEMO](https://blog.csdn.net/gangzhucoll/article/details/78516292)

[【数字图像处理系列一】opencv-python快速入门篇](https://zhuanlan.zhihu.com/p/44255577)