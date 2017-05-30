# vim:fileencoding=utf-8

import numpy as np
from numpy.random import *

import sys

import csv

def getNearestValue(list, num):

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    tmp = np.asarray(list) - num
    tmp[tmp > 0.0] = sys.float_info.min
    idx = tmp.argmax()

    return idx

def createCSVForDataset(responseFile, bodyFile, lookFile):
    responseCSV = open(responseFile, newline='')
    bodyCSV = open(bodyFile, newline='')
    lookCSV = open(lookFile, newline='')

    responseLines = csv.reader(responseCSV, delimiter=',')
    bodyLines = csv.reader(bodyCSV, delimiter=',')
    lookLines = csv.reader(lookCSV, delimiter=',')

    # 教師データ
    body_ans = np.loadtxt(bodyFile, delimiter = ',')
    look_ans = np.loadtxt(lookFile, delimiter = ',')

    time = np.empty((0, 1))

    body_y = np.empty((0, 2))
    look_y = np.empty((0, 2))

    # 入力データ
    body_X = np.empty((0, 3))
    look_X = np.empty((0, 2))

    for item in responseLines:
        body_X = np.append(body_X, np.array([[float(item[1]), float(item[2]), bool(item[3])]]), axis = 0)
        look_X = np.append(look_X, np.array([[float(item[4]), float(item[6])]]), axis = 0)

        time = np.append(time, np.array([[int(item[0])]]), axis = 0)

        idx = getNearestValue(body_ans[:,0], int(item[0]))
        val = body_ans[idx][1]
        body_y = np.append(body_y, np.array([[val, 1 - val]]), axis = 0)

        idx = getNearestValue(look_ans[:,0], int(item[0]))
        val = look_ans[idx][1]
        look_y = np.append(look_y, np.array([[val, 1 - val]]), axis = 0)

    dataFormat = np.concatenate((time, body_X, body_y, look_X, look_y), axis = 1)

    np.savetxt("dataset/dataset.csv", dataFormat, delimiter=",", header="time,heartrate,emg,emgEnable,bodyTrue,bodyFalse,voice,face,lookTrue,lookFalse")

    print(dataFormat)

def joinCSV(*filenames, output="datasets.csv"):
    result = np.empty((0, 10))
    fileList = list(filenames)
    for filename in fileList:
        dataset = np.loadtxt(filename, delimiter=",", skiprows=1)
        result = np.append(result, dataset, axis = 0)

    print(result)
    np.savetxt(output, result, delimiter=",", header="time,heartrate,emg,emgEnable,bodyTrue,bodyFalse,voice,face,lookTrue,lookFalse")
