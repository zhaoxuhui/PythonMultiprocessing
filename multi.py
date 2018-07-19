# coding: utf-8
from multiprocessing import Pool
import time
import os
import cv2
import numpy as np


def findAllFiles(root_dir, filter):
    print("Finding files ends with \'" + filter + "\' ...")
    separator = os.path.sep
    paths = []
    names = []
    files = []
    # 遍历
    for parent, dirname, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(filter):
                paths.append(parent + separator)
                names.append(filename)
    for i in range(paths.__len__()):
        files.append(paths[i] + names[i])
    print (names.__len__().__str__() + " files have been found.")
    paths.sort()
    names.sort()
    files.sort()
    return paths, names, files


def match_SURF(img_pair):
    t1 = time.time()

    img1 = img_pair[0]
    img2 = img_pair[1]

    # 新建SURF对象，参数默认
    surf = cv2.xfeatures2d_SURF.create()
    # 调用函数进行SURF提取
    kp1, des1 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img1, None)
    kp2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(surf, img2, None)

    good_matches = []
    good_kps1 = []
    good_kps2 = []
    good_out_kp1 = []
    good_out_kp2 = []

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good_matches.append(matches[i])
            good_kps1.append(kp1[matches[i][0].queryIdx])
            good_kps2.append(kp2[matches[i][0].trainIdx])

    # surf匹配出来的点对都是KeyPoint类型的对象，所以需要解析一下才可以使用
    for i in range(good_kps1.__len__()):
        good_out_kp1.append([good_kps1[i].pt[0], good_kps1[i].pt[1]])
        good_out_kp2.append([good_kps2[i].pt[0], good_kps2[i].pt[1]])
    affine, mask = cv2.estimateAffine2D(np.array(good_out_kp2), np.array(good_out_kp1))
    img_resampled = cv2.warpAffine(img2, affine, (img1.shape[1], img1.shape[0]))
    t2 = time.time()
    print("kp1 size:" + kp1.__len__().__str__() + " kp2 size:" + kp2.__len__().__str__())
    print(affine)
    print("cost time:" + (t2 - t1).__str__())
    return affine, img_resampled


if __name__ == '__main__':
    isMulti = False
    MultiNum = 6

    res = []
    resample_imgs = []
    paths, names, files = findAllFiles("img", '.jpg')
    # 注意不要在循环里载入影像，这样会开辟出许多重复的空间
    # 先载入，然后添加则是浅拷贝，只相当于拷贝了内存的地址
    base_img = cv2.imread(files[0])
    for i in range(1, files.__len__()):
        resample_imgs.append([base_img, cv2.imread(files[i])])
        print(files[i] + " was loaded." + (i + 1).__str__() + "/" + files.__len__().__str__())

    # 多线程
    if isMulti:
        print("MultiProcess:" + MultiNum.__str__())
        pool = Pool(processes=MultiNum)
        t1 = time.time()
        # 注意map函数中传入的参数应该是可迭代对象，如list
        res = pool.map(match_SURF, resample_imgs)
        pool.close()
        pool.join()
        t2 = time.time()
        print("Total time:" + (t2 - t1).__str__())
    # 单线程
    else:
        print("SingleProcess")
        t1 = time.time()
        for i in range(resample_imgs.__len__()):
            print("\n")
            result = match_SURF(resample_imgs[i])
            res.append(result)
            print("processing " + (i + 1).__str__() + "/" + resample_imgs.__len__().__str__())
        t2 = time.time()
        print("\nTotal time:" + (t2 - t1).__str__() + "\n")

    # 保存重采后的影像
    for i in range(res.__len__()):
        cv2.imwrite("resampled/resampled_" + (i + 1).__str__().zfill(2) + ".jpg", res[i][1])
        print("saved " + (i + 1).__str__() + "/" + res.__len__().__str__())
