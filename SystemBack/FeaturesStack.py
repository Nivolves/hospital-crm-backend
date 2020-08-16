import numpy as np
import pandas as pd
import math
import cv2
from scipy.stats import skew
from scipy.stats import kurtosis as kurt
from PIL import Image


def normal_equations_2d(y, x):
    xtx = [[0, 0], [0, 0]]
    for i in range(len(x)):
        xtx[0][1] += x[i]
        xtx[0][0] += x[i] * x[i]
    xtx[1][0] = xtx[0][1]
    xtx[1][1] = len(x)
    xtxInv = [[0, 0], [0, 0]]
    d = 1 / (xtx[0][0] * xtx[1][1] - xtx[1][0] * xtx[0][1])
    xtxInv[0][0] = xtx[1][1] * d
    xtxInv[0][1] = -xtx[0][1] * d
    xtxInv[1][0] = -xtx[1][0] * d
    xtxInv[1][1] = xtx[0][0] * d
    xtxInvxt = [[0 for _ in range(len(x))], [0 for _ in range(len(x))]]
    for i in range(2):
        for j in range(len(x)):
            xtxInvxt[i][j] = xtxInv[i][0] * x[j] + xtxInv[i][1]
    theta = [0, 0]
    for i in range(2):
        for j in range(len(x)):
            theta[i] += xtxInvxt[i][j] * y[j]
    return theta


def bow_counting_dimension(image, startSize, finishSize, step=1):
    image_f = Image.open(image)
    image = Image.new("RGB", image_f.size)
    image.paste(image_f)
    baList = dict()
    bw = image.load()
    for b in range(startSize, finishSize + 1, step):
        hCount = int(image.size[1] / b)
        wCount = int(image.size[0] / b)
        filledBoxes = [[False for _ in range(hCount + (1 if (image.size[1] > hCount * b) else 0))] for _ in
                       range(wCount + (1 if (image.size[0] > wCount * b) else 0))]
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                if bw[x, y] != (255, 255, 255):
                    xBox = x / b
                    yBox = y / b
                    filledBoxes[int(xBox)][int(yBox)] = True
        a = 0
        for i in range(len(filledBoxes)):
            for j in range(len(filledBoxes[0])):
                if filledBoxes[i][j]:
                    a += 1
        baList.update({math.log(1 / b): math.log(a)})
    return baList


def mink_val(img):
    start_size = 1
    finish_size = 400
    step = 1
    baList = bow_counting_dimension(img, start_size, finish_size, step)
    lb = len(baList)
    y = [0 for _ in range(lb)]
    x = [0 for _ in range(lb)]
    c = 0
    for key in baList.keys():
        y[c] = baList[key]
        x[c] = key
        c += 1
    theta = normal_equations_2d(y, x)
    val = theta
    return val[0]


def get_greyscale_matrix(filename):
    matrix = cv2.imread(filename, 0)
    return matrix.astype(int)


# Difference matrix
def get_diff_matrix(orig_matrix, var):
    diff_matrix = []
    # horizontal differentiation
    if var == 'hor':
        for i in range(orig_matrix.shape[0]):
            row = []
            for j in range(orig_matrix.shape[1] - 1):
                row.append(orig_matrix[i][j + 1] - orig_matrix[i][j])
            diff_matrix.append(row)
    # vertical differentiation
    else:
        for i in range(orig_matrix.shape[0] - 1):
            row = []
            for j in range(orig_matrix.shape[1]):
                row.append(orig_matrix[i + 1][j] - orig_matrix[i][j])
            diff_matrix.append(row)

    diff_matrix = np.asarray(diff_matrix)
    diff_matrix += abs(np.amin(diff_matrix))
    return diff_matrix


# Grey Level Co-Occurrence Matrix
def get_glcm(matrix):
    a1 = matrix[:-1]
    a2 = matrix[1:]
    str_list = []
    for i in range(a1.size):
        str_list.append(str(a1[i]) + str(a2[i]))
    a3 = []
    for i in range(len(str_list)):
        a3.append(str_list.count(str_list[i]))
    glcm = pd.DataFrame({'x': a1, 'y': a2, 'z': a3})
    glcm = glcm.drop_duplicates()
    return glcm.sort_values(['x', 'y'])


# Get GLCM features
def get_x1x2x3(glcm, image_features, best_grad1, best_grad2):
    a1 = glcm['x'].tolist()
    a2 = glcm['y'].tolist()
    a3 = glcm['z'].tolist()
    max_index = a3.index(np.amax(a3))
    result = a1[max_index] * a2[max_index]
    image_features.append(result)  # x1
    norm_value = pd.Series(a3)
    norm_value = (norm_value / sum(norm_value) * 10000).tolist()
    tp = np.array(a1)
    arr_index = np.where(tp == best_grad1)[0]
    result_array = []
    for j in arr_index:
        result_array.append(norm_value[j])
    try:
        result = sum(result_array) / len(result_array)
    except:
        result = 0
    image_features.append(result)  # x2
    arr_index = np.where(tp == best_grad2)[0]
    result_array = []
    for j in arr_index:
        result_array.append(norm_value[j])
    try:
        result = sum(result_array) / len(result_array)
    except:
        result = 0
    image_features.append(result)  # balx2
    result = np.amax(a1) * np.amax(a2)
    image_features.append(result)  # x3
    return image_features


# Grey Level Run Length Matrix
def get_glrlm(diff_matrix):
    glrlm = np.asarray(np.zeros((np.amax(diff_matrix) + 1, 3)), int)
    for i in range(3):
        d = i % 3
        for j in range(diff_matrix.size - d):
            if i == 0:
                glrlm[diff_matrix[j]][i] += 1
            elif i == 1:
                if diff_matrix[j] == diff_matrix[j + 1]:
                    glrlm[diff_matrix[j]][i] += 1
            else:
                if diff_matrix[j] == diff_matrix[j + 1] == diff_matrix[j + 2]:
                    glrlm[diff_matrix[j]][i] += 1
    return pd.DataFrame(glrlm)


def get_max_features(row, best_pairs, glcm):
    for best_pair in best_pairs:
        pair = glcm[(glcm['x'] == best_pair[0]) & (glcm['y'] == best_pair[1])]
        row.append((np.sum(pair['z'].values) / np.sum(glcm['z'])) * 10000)
    return row


def get_norm_features(init_matrix, image_features, diff_type, best_grad1, best_grad2, best_pairs, flag=False):
    temp_diff_matrix = get_diff_matrix(init_matrix, diff_type)
    diff_matrix = np.concatenate(temp_diff_matrix, axis=None)
    image_features.append((np.sum(diff_matrix == np.amin(diff_matrix)) / diff_matrix.size) * 100)
    image_features.append((np.sum(diff_matrix == np.amax(diff_matrix)) / diff_matrix.size) * 100)

    # glcm
    glcm = get_glcm(diff_matrix)
    image_features = get_x1x2x3(glcm, image_features, best_grad1, best_grad2)

    # statistical features
    image_features.append(np.mean(diff_matrix))
    image_features.append(np.std(diff_matrix))
    image_features.append(skew(diff_matrix))
    image_features.append(kurt(diff_matrix))
    image_features.append(np.amax(diff_matrix) - np.amin(diff_matrix))  # range
    image_features.append(np.median(diff_matrix))
    Q1 = np.percentile(diff_matrix, 25, interpolation='midpoint')
    Q3 = np.percentile(diff_matrix, 75, interpolation='midpoint')
    image_features.append(Q1)
    image_features.append(Q3)
    image_features.append(np.percentile(diff_matrix, 5, interpolation='midpoint'))
    image_features.append(np.percentile(diff_matrix, 95, interpolation='midpoint'))
    IQR = Q3 - Q1
    image_features.append(IQR)

    # glrlm
    glrlm = get_glrlm(diff_matrix)
    # length = 1
    image_features.append(np.mean(glrlm[0]))
    image_features.append(np.std(glrlm[0]))
    image_features.append(skew(glrlm[0]))
    image_features.append(kurt(glrlm[0]))
    image_features.append(np.amax(glrlm[0]) - np.amin(glrlm[0]))  # range
    image_features.append(np.median(glrlm[0]))
    Q1 = np.percentile(glrlm[0], 25, interpolation='midpoint')
    Q3 = np.percentile(glrlm[0], 75, interpolation='midpoint')
    image_features.append(Q1)
    image_features.append(Q3)
    image_features.append(np.percentile(glrlm[0], 5, interpolation='midpoint'))
    image_features.append(np.percentile(glrlm[0], 95, interpolation='midpoint'))
    IQR = Q3 - Q1
    image_features.append(IQR)

    # length = 2
    image_features.append(np.mean(glrlm[1]))
    image_features.append(np.std(glrlm[1]))
    image_features.append(skew(glrlm[1]))
    image_features.append(kurt(glrlm[1]))
    image_features.append(np.amax(glrlm[1]) - np.amin(glrlm[1]))  # range
    image_features.append(np.median(glrlm[1]))
    Q1 = np.percentile(glrlm[1], 25, interpolation='midpoint')
    Q3 = np.percentile(glrlm[1], 75, interpolation='midpoint')
    image_features.append(Q1)
    image_features.append(Q3)
    image_features.append(np.percentile(glrlm[1], 95, interpolation='midpoint'))
    IQR = Q3 - Q1
    image_features.append(IQR)

    # length = 3
    image_features.append(np.mean(glrlm[2]))
    image_features.append(np.std(glrlm[2]))
    image_features.append(skew(glrlm[2]))
    image_features.append(kurt(glrlm[2]))
    image_features.append(np.amax(glrlm[2]) - np.amin(glrlm[2]))  # range
    image_features.append(np.median(glrlm[2]))
    image_features.append(np.percentile(glrlm[2], 75, interpolation='midpoint'))  # Q3
    image_features.append(np.percentile(glrlm[2], 95, interpolation='midpoint'))

    # differences of mode amplitudes
    glrlm = glrlm.div(diff_matrix.size)
    glrlm = glrlm.multiply(100)
    image_features.append(np.amax(glrlm[0]) - np.amax(glrlm[1]))  # dif12
    image_features.append(np.amax(glrlm[0]) - np.amax(glrlm[2]))  # dif13
    image_features.append(np.amax(glrlm[1]) - np.amax(glrlm[2]))  # dif23

    # Max Goncharuk features
    image_features = get_max_features(image_features, best_pairs, glcm)

    if flag:
        return image_features, temp_diff_matrix
    else:
        return image_features
