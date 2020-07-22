import numpy as np
import cv2 as cv
import glob

# 终止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 准备对象点， 如 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# 用于存储所有图像的对象点和图像点的数组。
objpoints = []  # 真实世界中的3d点
imgpoints = []  # 图像中的2d点
images = glob.glob('pictures/left/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 找到棋盘角落
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # 如果找到，添加对象点，图像点（细化之后）
    if ret == True:
        size = gray.shape[::-1]
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # 绘制并显示拐角
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)
cv.destroyAllWindows()

ret, mtx, dist, r_vecs, t_vecs = cv.calibrateCamera(objpoints, imgpoints, size, None, None)



for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 找到棋盘角落
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    # 如果找到，添加对象点，图像点（细化之后）
    if ret == True:
        size = gray.shape[::-1]
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # 绘制并显示拐角
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)
cv.destroyAllWindows()


#
# img = cv.imread('pictures/left/left05.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
#
# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # 剪裁图像
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
# cv.imwrite('calibrate_result.png', dst)
