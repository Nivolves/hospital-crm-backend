# coding=utf-8
import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis
import copy
import sys
from math import log, pow, sqrt, sin, cos, atan
import time
import json


# Матрица градаций серого
def getPixelMatrix(filename):
    arr = cv2.imread(filename, 0)
    arr = arr.astype(int)
    a = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            a.append(arr[i][j])
    return a


# Нормировка матрицы
def getNormArr(a):
    normArr = []
    for i in range(len(a) - 1):
        z = a[i + 1] - a[i]
        normArr.append(z)
    minArr = abs(min(normArr))
    for i in range(len(normArr)):
        normArr[i] += minArr
    return normArr


# Матрица сочетаний
def getGLCM(arr):
    a1 = copy.deepcopy(arr)
    a2 = []
    for i in range(1, len(a1)):
        a2.append(a1[i])
    del a1[-1]
    strA = []
    for i in range(len(a1)):
        strA.append(str(a1[i]) + str(a2[i]))
    a3 = []
    for i in range(len(strA)):
        a3.append(strA.count(strA[i]))
    df = pd.DataFrame({'x': a1, 'y': a2, 'z': a3})
    df = df.drop_duplicates()
    return df.sort_values(by=['x'])


# Матрица длин градаций серого
def getGLRLM(normArr):
    minArr = abs(min(normArr))
    for i in range(len(normArr)):
        normArr[i] += minArr
    maxArr = max(normArr) + 1
    zArr = np.zeros((maxArr, 3))
    zArr = zArr.astype(int)
    for i in range(3):
        d = i % 3
        for j in range(len(normArr)):
            if (j != (len(normArr) - d)):
                if i == 0:
                    zArr[normArr[j]][i] += 1
                elif i == 1:
                    if (normArr[j] == normArr[j + 1]):
                        zArr[normArr[j]][i] += 1
                else:
                    if (normArr[j] == normArr[j + 1] == normArr[j + 2]):
                        zArr[normArr[j]][i] += 1
            else:
                break
    return pd.DataFrame(zArr)


# Первый признак матрицы GLCM
def getX1(glcm):
    a1 = glcm['x'].tolist()
    a2 = glcm['y'].tolist()
    a3 = glcm['z'].tolist()
    max_index = a3.index(max(a3))
    x1 = int(a1[max_index]) * int(a2[max_index])
    return x1


# Второй признак матрицы GLCM
def getX2(glcm):
    a1 = glcm['x'].tolist()
    a3 = glcm['z'].tolist()
    norm_value = pd.Series(a3)
    norm_value = (norm_value / sum(norm_value) * 10000).tolist()
    tp = np.array(a1)
    arr_index = np.where(tp == 53)[0]
    result2_array = []
    for j in arr_index:
        result2_array.append(norm_value[j])
    try:
        x2 = sum(result2_array) / len(result2_array)
    except:
        x2 = 0
    return x2


# Третий признак матрицы GLCM
def getX3(glcm):
    a1 = glcm['x'].tolist()
    a2 = glcm['y'].tolist()
    x3 = int(max(a1)) * int(max(a2))
    return x3


# Среднее значение
def getMean(arr):
    return np.mean(arr)


# Медиана
def getMedian(arr):
    return np.median(arr)


# Первый квартиль
def getQ1(arr):
    return np.percentile(arr, 25, interpolation='midpoint')


# Третий квартиль
def getQ3(arr):
    return np.percentile(arr, 75, interpolation='midpoint')


# Межквартильный размах
def getIQR(arr):
    return (getQ3(arr) - getQ1(arr))


# Стандартное отклонение
def getStd(arr):
    return np.std(arr)


# Ассиметрия
def getSkew(arr):
    return skew(arr)


# Эксцесс
def getKurtosis(arr):
    return kurtosis(arr)


# 5% квартиль
def getQ5(arr):
    return np.percentile(arr, 5, interpolation='midpoint')


# 95% квартиль
def getQ95(arr):
    return np.percentile(arr, 95, interpolation='midpoint')


# Размах
def getRange(arr):
    return (max(arr) - min(arr))


# Разница между амплитудами мод длин 1 и 2
def getDif12(glrlm):
    glrlm = glrlm.div(len(arr))
    glrlm = glrlm.multiply(100)
    return (max(glrlm[0]) - max(glrlm[1]))


# Разница между амплитудами мод длин 2 и 3
def getDif23(glrlm):
    glrlm = glrlm.div(len(arr))
    glrlm = glrlm.multiply(100)
    return (max(glrlm[1]) - max(glrlm[2]))


# Разница между амплитудами мод длин 1 и 3
def getDif13(glrlm):
    glrlm = glrlm.div(len(arr))
    glrlm = glrlm.multiply(100)
    return (max(glrlm[0]) - max(glrlm[2]))


# Модель для конвексного датчика
def convexModel():
    rrange = getRange(normArr)
    dif12 = getDif12(glrlm=glrlm)
    x3 = getX3(glcm=glcm)
    std = getStd(normArr)
    x3_norm = getX3(normGLCM)
    dif13 = getDif13(glrlm=glrlm)
    IQR = getIQR(normArr)
    Q95 = getQ95(normArr)
    result = 12.7436 + (0.359819 * sqrt(rrange)) / log(dif12) + (0.00470603 * x3) / pow(std, 3) - \
             4.07882e-19 * x3_norm * pow(x3, 3) + (566923000 * pow(std, 3)) / pow(x3, 3) - \
             (3767.95 * x3_norm) / pow(x3, 2) - 0.255481 * sin(dif13) * cos(std) + \
             (44.8881 * pow(IQR, 3)) / pow(Q95, 3) - (1.59326 * pow(std, 3)) / pow(Q95, 2) - \
             (474.325 * sin(dif12)) / x3_norm - (14.2729 * atan(dif12)) / atan(dif13)
    return 2 if result > 0.5 else 1


# Модель для линейного датчика
def linearModel():
    x1 = getX1(glcm=glcm)
    x3_norm = getX3(normGLCM)
    std = getStd(normArr)
    skew_1 = getSkew(glrlm[0])
    range_1 = getRange(glrlm[0])
    Q3_2 = getQ3(glrlm[1])
    result = -0.45995 + 0.000143428 * x1 - 0.0000151906 * x3_norm - 0.00236257 * std + 0.323126 * skew_1 + \
             0.000071245 * range_1 + 0.0146191 * Q3_2
    return 2 if result > 0.5 else 1


# Модель для линейного датчика в усиленном режиме (яичко)
def ballsModel():
    x1 = getX1(glcm=glcm)
    median = getMedian(normArr)
    Q5 = getQ5(normArr)
    Q95 = getQ95(normArr)
    IQR_1 = getIQR(glrlm[0])
    result = 1.33614 + 0.0000643571 * x1 + 0.477101 * median - 0.202878 * Q5 - 0.280842 * Q95 - 0.00119196 * IQR_1
    return 2 if result > 0.5 else 1


# def getTypeResult(type_sensor):
#     type_result = []
#     if res == 1:
#             return "Печень в норме"
#         type_result.append("Линейная модель: норма") if type_sensor == 1 \
#             else type_result.append("Конвексная модель: норма")
#     else:
#         type_result.append("Линейная модель: патология") if type_sensor == 1 \
#             else type_result.append("Конвексная модель: патология")
#     return type_result


def getMeanSigns(type_sensor):
    if type_sensor == "linear":
        x1 = getX1(glcm=glcm)
        skew_1 = getSkew(glrlm[0])
        range_1 = getRange(glrlm[0])

        return [['x1', int(1444 / (max(1444, x1) * 1.1) * 100), int(x1 / (max(1444, x1) * 1.10) * 100)],
                ['skew(1)', int(2 / (max(2, skew_1) * 1.2) * 100), int(skew_1 / (max(2, skew_1) * 1.2) * 100)],
                # ассиметрия частот градаций серого с длиной 1
                ['range(1)', int(928 / (max(928, range_1) * 1.3) * 100),
                 int(range_1 / (max(928, range_1) * 1.3) * 100)]]  # размах частот градаций серого с длиной 1

    elif type_sensor == "reinforced_linear":
        median = getMedian(normArr)
        Q5 = getQ5(normArr)
        Q95 = getQ95(normArr)
        return [['median', int(27 / (max(27, median) * 1.1) * 100), int(median / (max(27, median) * 1.10) * 100)],
                # медиана градаций серого
                ['Q5', int(27 / (max(27, Q5) * 1.2) * 100), int(Q5 / (max(27, Q5) * 1.2) * 100)],
                # 5% квартиль градаций серого
                ['Q95', int(33 / (max(33, Q95) * 1.3) * 100),
                 int(Q95 / (max(33, Q95) * 1.3) * 100)]]  # 95% квартиль градаций серого
    else:
        x3 = getX3(glcm=glcm)
        std = getStd(normArr)
        IQR = getIQR(normArr)
        return [['cube(std)', int(307 / (max(307, pow(std, 3)) * 1.1) * 100),
                 int(pow(std, 3) / (max(307, pow(std, 3)) * 1.10) * 100)],
                # медиана градаций серого
                ['cube(IQR)', int(512 / (max(512, pow(IQR, 3)) * 1.2) * 100),
                 int(pow(IQR, 3) / (max(512, pow(IQR, 3)) * 1.2) * 100)],
                # 5% квартиль градаций серого
                ['x3', int(18769 / (max(18769, x3) * 1.3) * 100),
                 int(x3 / (max(18769, x3) * 1.3) * 100)]]  # 95% квартиль градаций серого


def main(path, type_task, type_sensor):
    global arr, normArr, glcm, normGLCM, glrlm, res
    arr = getPixelMatrix(path)  # матрица градаций серого
    normArr = getNormArr(arr)  # нормализированная матрица
    glcm = getGLCM(arr)  # GLCM
    normGLCM = getGLCM(normArr)  # нормализированная GLCM
    glrlm = getGLRLM(normArr)  # GLRLM
    # type_task: 1 - норма/патология, 2 - стадия фиброза
    if type_task == "isNormal":
        # type_sensor: 1 - линейный датчик, 2 - линейный датчик в усиленном режиме (яичко), 3 - конвексный датчик
        if type_sensor == "linear":
            res = linearModel()
        elif type_sensor == "reinforced_linear":
            res = ballsModel()
        else:
            res = convexModel()
    type_result = "Печень в норме" if res == 1 else "Печень не в норме"
    mean_signs = getMeanSigns(type_sensor)
    print(json.dumps({"type_result": type_result, "mean_signs": mean_signs}))


if __name__ == '__main__':
    link = sys.argv[1]
    type_task = sys.argv[2]
    type_sensor = sys.argv[3]
    main(link, type_task, type_sensor)

