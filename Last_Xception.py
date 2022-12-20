import pandas as pd

import tensorflow as tf

from util import get_data_newfour
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.callbacks import EarlyStopping
import keras_metrics as km


import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization, GlobalMaxPooling2D
import warnings
warnings.filterwarnings("ignore")

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':
    train_ds, val_ds = get_data_newfour()

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    # 初始化DenseNet169网络(卷积神经网络的一种)
    mobile_net = Xception(input_shape=(300, 300, 3), include_top=False)
    # 固定参数
    mobile_net.trainable = False

    model = Sequential([
        mobile_net,
        GlobalMaxPooling2D(),
        Flatten(),
        Dense(1000, activation='relu'),
        BatchNormalization(),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(2, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', km.f1_score(), km.recall(), km.precision()])

    early_stopping = EarlyStopping(

        monitor='val_accuracy',
        verbose=1,
        patience=20,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(min_lr=0.00001,
                                                     factor=0.2)

    history = model.fit(train_ds,steps_per_epoch=len(train_ds), epochs=1000, callbacks=[early_stopping, reduce_lr], validation_data=val_ds, validation_steps=len(val_ds))

    predict = []

    val_targ = []
    for i in range(14):
        res = model.predict(val_ds[i][0])
        [predict.append(np.argmax(r)) for r in res]
        [val_targ.append(np.argmax(label)) for label in val_ds[i][1]]

    print(len(predict))
    print(len(val_targ))
    print(predict)
    print(val_targ)

    _val_f1 = f1_score(val_targ, predict, average='micro')
    _val_recall = recall_score(val_targ, predict, average=None)  ###
    _val_precision = precision_score(val_targ, predict, average=None)  ###
    print('_val_f1', _val_f1)
    print('_val_recall', _val_recall[0])
    print('_val_precision', _val_precision[0])

    hist_df = pd.DataFrame(history.history)

    y_pre_file = 'base3_Xception_predict.csv'
    y_rel_file = 'base3_Xception_reltag.csv'
    test1 = pd.DataFrame(data=predict)
    test2 = pd.DataFrame(data=val_targ)
    test1.to_csv(y_pre_file, encoding='utf-8')
    test2.to_csv(y_rel_file, encoding='utf-8')

    hist_csv_file = 'base3_Xception_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    # model.save('base3_Xception.h5')
