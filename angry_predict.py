# vim:fileencoding=utf-8

import angry_look_predict as look
import angry_body_predict as body

import numpy as np
from numpy.random import *

import sys

import csv

# 怒りを判定
def angry_predict(heartrate, myo, myoEnable, face, faceEnable, voice, voiceEnable):
    bodyInput = np.array([[heartrate, myo, myoEnable]])
    lookInput = np.array([[face, voice]])

    bodyResult = body.predict(bodyInput)
    lookResult = look.predict(lookInput)

    return bodyResult[0][0], lookResult[0][0]

def angry_fit(filename):

    datasets = np.loadtxt(filename, delimiter=",", skiprows=1)

    body_X = datasets[:,1:4]
    body_y = datasets[:,4:6]
    look_X = datasets[:,6:8]
    look_y = datasets[:,8:10]

    # responseCSV = open(responseFile, newline='')
    # bodyCSV = open(bodyFile, newline='')
    # lookCSV = open(lookFile, newline='')
    #
    # responseLines = csv.reader(responseCSV, delimiter=',')
    # bodyLines = csv.reader(bodyCSV, delimiter=',')
    # lookLines = csv.reader(lookCSV, delimiter=',')
    #
    # # 教師データ
    # body_ans = np.loadtxt(bodyFile, delimiter = ',')
    # look_ans = np.loadtxt(lookFile, delimiter = ',')
    #
    # print(body_ans)
    # print(look_ans)
    #
    # body_y = np.empty((0, 2))
    # look_y = np.empty((0, 2))
    #
    # # 入力データ
    # body_X = np.empty((0, 3))
    # look_X = np.empty((0, 2))
    #
    # for item in responseLines:
    #     body_X = np.append(body_X, np.array([[float(item[1]), float(item[2]), bool(item[3])]]), axis = 0)
    #     look_X = np.append(look_X, np.array([[float(item[4]), float(item[6])]]), axis = 0)
    #
    #     idx = getNearestValue(body_ans[:,0], int(item[0]))
    #     val = body_ans[idx][1]
    #     body_y = np.append(body_y, np.array([[val, 1 - val]]), axis = 0)
    #
    #     idx = getNearestValue(look_ans[:,0], int(item[0]))
    #     val = look_ans[idx][1]
    #     look_y = np.append(look_y, np.array([[val, 1 - val]]), axis = 0)

    print(body_X)
    print(look_X)

    print(body_y)
    print(look_y)

    body.fit(body_X, body_y)
    look.fit(look_X, look_y)


if __name__ == '__main__':
    angry_fit("responseAngry.csv", "body.csv", "look.csv")

    bodyResult, lookResult = angry_predict(0.2, 0.3, True, 0.7, True, 0.8, True)

    print(bodyResult)
    print(lookResult)
