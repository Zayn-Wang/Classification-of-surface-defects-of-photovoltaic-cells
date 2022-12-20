import pandas as pd

import tensorflow as tf

from util import get_data_newfour
from tensorflow.python.keras.callbacks import EarlyStopping
import keras_metrics as km
import keras
from keras import backend as K

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# def exponent(global_epoch,
#             learning_rate_base,
#             decay_rate,
#             min_learn_rate=0,
#             ):
#
#     learning_rate = learning_rate_base * pow(decay_rate, global_epoch)
#     learning_rate = max(learning_rate,min_learn_rate)
#     return learning_rate
#
# class ExponentDecayScheduler(keras.callbacks.Callback):
#     """
#     继承Callback，实现对学习率的调度
#     """
#     def __init__(self,
#                  learning_rate_base,
#                  decay_rate,
#                  global_epoch_init=0,
#                  min_learn_rate=0,
#                  verbose=0):
#         super(ExponentDecayScheduler, self).__init__()
#         # 基础的学习率
#         self.learning_rate_base = learning_rate_base
#         # 全局初始化epoch
#         self.global_epoch = global_epoch_init
#
#         self.decay_rate = decay_rate
#         # 参数显示
#         self.verbose = verbose
#         # learning_rates用于记录每次更新后的学习率，方便图形化观察
#         self.min_learn_rate = min_learn_rate
#         self.learning_rates = []
#
#     def on_epoch_end(self, epochs ,logs=None):
#         self.global_epoch = self.global_epoch + 1
#         lr = K.get_value(self.model.optimizer.lr)
#         self.learning_rates.append(lr)
# 	#更新学习率
#     def on_epoch_begin(self, batch, logs=None):
#         lr = exponent(global_epoch=self.global_epoch,
#                     learning_rate_base=self.learning_rate_base,
#                     decay_rate = self.decay_rate,
#                     min_learn_rate = self.min_learn_rate)
#         K.set_value(self.model.optimizer.lr, lr)
#         if self.verbose > 0:
#             print('\nBatch %05d: setting learning '
#                   'rate to %s.' % (self.global_epoch + 1, lr))

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':
    train_ds, val_ds = get_data_newfour()

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (10, 10), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

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

    y_pre_file = 'base8_cnn_predict.csv'
    y_rel_file = 'base8_cnn_reltag.csv'
    test1 = pd.DataFrame(data=predict)
    test2 = pd.DataFrame(data=val_targ)
    test1.to_csv(y_pre_file, encoding='utf-8')
    test2.to_csv(y_rel_file, encoding='utf-8')

    hist_csv_file = 'base8_cnn_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    # model.save('base8_cnn.h5')
