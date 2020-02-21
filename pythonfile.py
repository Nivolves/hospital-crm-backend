# coding=utf-8
import numpy as np
import pandas as pd
from cv2 import cv2
import sys
from scipy.stats import skew, kurtosis
from datetime import timedelta
import copy
from math import log, pow, sqrt, sin, cos, atan
from flask_cors import CORS, cross_origin
from flask import Flask, redirect, url_for, request
from functools import update_wrapper
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


def getPixelMatrix(filename):
    arr = cv2.imread(filename, 0)
    return arr.astype(int)


def getNormArr(a):
    normArr = []
    for i in range(len(a) - 1):
        z = a[i + 1] - a[i]
        normArr.append(z)
    minArr = abs(min(normArr))
    for i in range(len(normArr)):
        normArr[i] += minArr
    return normArr


def getGLCM(arr):
    a1 = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            a1.append(arr[i][j])
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


def getNormGLCM(normArr):
    a1 = copy.deepcopy(normArr)
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


def getX1X2X3(df):
    newList = []
    a1 = df['x'].tolist()
    a2 = df['y'].tolist()
    a3 = df['z'].tolist()
    fin_matrix = []
    for i in range(len(a1)):
        fin_matrix.append([a1[i], a2[i], a3[i]])
    final_matrix = np.zeros((max(a1 + a2) + 1, max(a1 + a2) + 1), dtype=int)
    for i in fin_matrix:
        final_matrix[i[0]][i[1]] = i[2]
    max_index = a3.index(max(a3))
    result = int(a1[max_index]) * int(a2[max_index])
    newList.append(result)  # первый признак (x1)
    norm_value = pd.Series(a3)
    norm_value = (norm_value / sum(norm_value) * 10000).tolist()
    tp = np.array(a1)
    arr_index = np.where(tp == 53)[0]
    result2_array = []
    for j in arr_index:
        result2_array.append(norm_value[j])
    try:
        result2 = sum(result2_array) / len(result2_array)
    except:
        result2 = 0
    newList.append(result2)  # второй признак (x2)
    result3 = int(max(a1)) * int(max(a2))
    newList.append(result3)  # третий признак (x3)
    return newList


def convexModel():
    result = 10.9313 - 0.0000113586 * x3 * cos(skew2) - (10.2813 * atan(Q95_1)) / atan(mean1) - 0.00190833 * sqrt(
        x1) * sin(dif13) + \
        (0.00000191699 * x1) / sin(dif13) + 12.0079 / (x1 * log(kurtosis2)) + (0.0000000463565 * pow(Q3, 3)) / cos(
        mean3) + \
        (24.1914 * pow(std, 3)) / pow(Q95, 3) - (0.244841 * pow(x3, 1. / 3)) / sqrt(x1) - (
        0.829612 * pow(Q95, 2)) / x3 - \
        0.00418883 * pow(x3, 1. / 3) * cos(mean3) + (24337900 * sin(dif13)) / pow(x3, 2) + (
        0.0000825577 * pow(Q95, 2)) / pow(kurtosis2, 1. / 3) - \
        (14.62 * cos(skew2)) / sqrt(x3) - (361.806 * sin(dif13)) / x1norm + 814.946 * sin(dif13) / pow(Q95, 2) - \
        331.19 / pow(x3, 5. / 6) - (3428.16 * sin(dif13)) / x3 + \
        0.0000000396005 * pow(median, 2) * pow(std, 3)
    return 2 if result > 0.5 else 1


def linearModel():
    result = -0.476814 + (0.508205 * log(x1)) / log(std) - (8.8652 * log(x1)) / sqrt(x3) - (
        7.02162 * pow(mean3, 1. / 3)) / sqrt(mean2) \
        + (0.000211017 * sqrt(x1)) / cos(skew1) - 0.00594114 * sqrt(x3) * log(skew1) - 0.0148594 / (
        pow(range2, 1. / 3) * cos(skew1)) \
        + (4.52131 * pow(std3, 1. / 3)) / sqrt(mean2) - 7.05488 / (log(x3) * atan(skew1)) + (
        5.84287 * atan(Q95_3)) / pow(range1, 1. / 3) + \
        0.00424622 * sqrt(std2) * pow(range2, 1. / 3) - (1.17722 * sqrt(mean3)) / std + (
        5.16211 * atan(dif23)) / pow(x1, 1. / 3) \
        - (5.82559 * atan(std3)) / pow(Q95_1, 1. / 3) + 0.00285946 * pow(x3 * Q95_3, 1. / 3) + \
        0.0203011 * pow(x1, 1. / 3) * atan(std3)
    return 2 if result > 0.5 else 1


def clModel():
    result = 0.378151 + 0.000000331558 * x1 * std1 + 0.000000219131 * x3 * std1 - 0.00000208405 * x3 * std2 + \
        0.0000000881623 * x3 * range2 - 0.00470088 * std1 + \
        0.0317531 * std2 + 0.00000109125 * std2 * range2
    return 2 if result > 0.5 else 1


def getTypeResult(type_sensor):
    type_result = []
    if firstRes == 1:
        type_result.append("Линейная модель: норма") if type_sensor == 1 \
            else type_result.append("Конвексная модель: норма")
    else:
        type_result.append("Линейная модель: патология") if type_sensor == 1 \
            else type_result.append("Конвексная модель: патология")
    type_result.append("Смешанная модель: норма") if secondRes == 1 \
        else type_result.append("Смешанная модель: патология")
    return type_result


def getMeanSigns(type_sensor):
    return [[['cubert(mean(3))', '1.36', str(pow(mean3, 1. / 3))],
             ['cubert(std(3))', '1.88', str(pow(std3, 1. / 3))],
             ['arctg(Q95(3))', '1.519', str(atan(Q95_3))]],
            [['std(2)', '28.65', str(std2),
              'std(1)', '186', str(std1),
              'range(2)', '80', str(range2)]]] \
        if type_sensor == 1 else \
        [[['x1', '5775', str(x1)],
          ['sqrt(x1)', '76', str(sqrt(x1))],
          ['ln(kurtosis(2)', '1.68', str(log(kurtosis2))]],
         [['std(2)', '28.65', str(std2),
           'std(1)', '186', str(std1),
           'range(2)', '80', str(range2)]]]


def getProcentResult(type_sensor):
    return ['Точность линейной модели: 83.1%', 'Точность смешанной модели: 80%'] if type_sensor == 1 \
        else ['Точность конвексной модели: 80.3%', 'Точность смешанной модели: 80%']


def main(path, type_task, type_sensor):
    global x1, x2, x3, x1norm, x2norm, x3norm, median, Q3, std, Q95, mean1, std1, skew1, Q95_1, range1, mean2, std2, \
        skew2, kurtosis2, range2, mean3, std3, Q95_3, dif23, dif13, firstRes, secondRes

    arr = getPixelMatrix(path)  # матрица градаций серого

    a = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            a.append(arr[i][j])
    normArr = getNormArr(a)  # нормализированная матрица

    df = getGLCM(arr)  # GLCM
    newList = getX1X2X3(df)
    x1, x2, x3 = newList[0], newList[1], newList[2]

    df = getNormGLCM(normArr)  # нормализированная GLCM
    newList = getX1X2X3(df)
    x1norm, x2norm, x3norm = newList[0], newList[1], newList[2]

    # матрица градаций серого
    median = np.median(normArr)  # медиана
    # третий квартиль
    Q3 = np.percentile(normArr, 75, interpolation='midpoint')
    std = np.std(normArr)  # стандартное отклонение
    Q95 = np.percentile(normArr, 95, interpolation='midpoint')  # 95% квартиль

    df = getGLRLM(normArr)  # GLRLM

    # матрица длин градаций серого
    # длина 1
    mean1 = np.mean(df[0])  # среднее значение
    std1 = np.std(df[0])  # стандартное отклонение
    skew1 = skew(df[0])  # ассиметрия
    Q95_1 = np.percentile(df[0], 95, interpolation='midpoint')  # 95% квартиль
    range1 = max(df[0]) - min(df[0])  # размах
    # длина 2
    mean2 = np.mean(df[1])  # среднее значение
    std2 = np.std(df[1])  # стандартное отклонение
    skew2 = skew(df[1])  # ассиметрия
    kurtosis2 = kurtosis(df[1])  # эксцесс
    range2 = max(df[1]) - min(df[1])  # размах
    # длина 3
    mean3 = np.mean(df[2])  # среднее значение
    std3 = np.std(df[2])  # стандартное отклонение
    Q95_3 = np.percentile(df[2], 95, interpolation='midpoint')  # 95% квартиль
    # разницы между амплитудами мод
    df = df.div(len(a))
    df = df.multiply(100)
    dif23 = max(df[1]) - max(df[2])  # 2 - 3
    dif13 = max(df[0]) - max(df[2])  # 1 - 3

    if type_task == "isNormal":
        firstRes = linearModel() if type_sensor == "linear" else convexModel()
        secondRes = clModel()
    type_result = getTypeResult(type_sensor)
    mean_signs = getMeanSigns(type_sensor)
    # procent_result = getProcentResult(type_sensor)
    print(json.dumps({"type_result": type_result, "mean_signs": mean_signs}))


if __name__ == '__main__':
    link = sys.argv[1]
    type_task = sys.argv[2]
    type_sensor = sys.argv[3]
    main(link, type_task, type_sensor)
