# ネットワーク構築
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation

# 学習プロセス
from keras.optimizers import SGD

# コールバック
from keras.callbacks import ModelCheckpoint

import numpy as np
from numpy.random import *


# 学習させる関数
def angry_fit(X_train, y_train, model_name = "angry.hdf5"):

    # 初期化
    model = Sequential()

    # 層の追加
    model.add(Dense(output_dim=3, input_dim=2))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=2))
    model.add(Activation("softmax"))

    # optimizer = "sgd"でも、簡単に最適化関数の設定ができる
    #(その場合パラメータはデフォルト値)
    model.compile(loss="categorical_crossentropy",
                  optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                  metrics=["accuracy"])

    # コールバックで学習済みモデルの保存
    check = ModelCheckpoint(model_name)

    # 学習開始
    history = model.fit(X_train, y_train, nb_epoch=10,
                        validation_split=0.2, batch_size=32,
                        callbacks=[check])

# ネットワークモデルを評価
def angry_evaluate(X_test, y_test, model_name = "angry.hdf5"):
    # モデル評価
    loss_and_metrics = model.evaluate(X_test,y_test,batch_size=32,verbose=0)
    print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))

# 推測
def angry_predict(X_data, model_name = "angry.hdf5"):
    # モデル読み込み
    model = load_model(model_name)

    y_result = model.predict(X_data)

    return y_result


if __name__ == '__main__':
    # テストデータ数 x 入力パラメータ数
    X_train = rand(10000, 2)

    y_true  = X_train[:, 0] + X_train[:, 1] >= 1.2
    y_false = y_true != True

    # 教師クラスデータ
    y_train = np.c_[y_true, y_false]

    # print(y_train)
    #
    # print(X_train.shape)
    # print(y_train.shape)

    angry_fit(X_train, y_train)

    X_test = rand(10, 2)

    y_true  = X_test[:, 0] + X_test[:, 1] >= 1.2
    y_false = y_true != True

    y_test = np.c_[y_true, y_false]

    y_result = angry_predict(X_test)

    print(X_test)

    print(y_test)
    print(y_result)
