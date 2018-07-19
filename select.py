# coding=utf-8
from multiprocessing import cpu_count
from multiprocessing import Pool
import time
import numpy as np


def processImg(img):
    # 利用Numpy矩阵操作对图像进行简单的二值化处理
    np.uint8(np.where(img < 128, 0, 255))


def takeTime(item):
    # 用于返回列表中指定位置的元素
    return item[1]


def selectCPUNum(imgNum=100, imgW=800, imgH=800):
    start_cpu_num = 1
    end_cpu_num = cpu_count()
    print("cpu num:" + cpu_count().__str__())
    print("test range:" + start_cpu_num.__str__() + " - " + end_cpu_num.__str__() + "\n")

    # 利用Numpy生成imgNum张imgW*imgH的随机矩阵作为图像用于测试
    imgs = []
    for i in range(imgNum):
        img = np.uint8(np.random.randint(0, 255, size=(imgW, imgH)))
        imgs.append(img)

    cost_time = []

    for MultiNum in range(start_cpu_num, end_cpu_num + 1):
        print("MultiProcess:" + MultiNum.__str__())
        pool = Pool(processes=MultiNum)
        t1 = time.time()
        pool.map(processImg, imgs)
        pool.close()
        pool.join()
        t2 = time.time()
        cost_time.append((MultiNum, t2 - t1))
        print("cost time:" + (t2 - t1).__str__())

    cost_time.sort(key=takeTime)
    print("\ntest result for num " + MultiNum.__str__() + ":")
    for item in cost_time:
        print(item)
    ave_num = int((cost_time[0][0] + cost_time[1][0] + cost_time[2][0]) / 3)
    print("\nrecommend cpu num:" + ave_num.__str__() + "\n")
    return ave_num


def selectCPUNumRobust(runTime=4, imgNum=100, imgW=800, imgH=800):
    nums = []
    for i in range(runTime):
        print("Run for " + (i + 1).__str__() + " time...")
        nums.append(selectCPUNum(imgNum=imgNum, imgW=imgW, imgH=imgH))

    final_num = sum(nums) / nums.__len__()
    print("final recommend cpu num:" + final_num.__str__())
    return final_num


if __name__ == '__main__':
    selectCPUNumRobust()
