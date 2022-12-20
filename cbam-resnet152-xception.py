from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.applications.xception import Xception

from tensorflow.python.keras.layers import Flatten, MaxPooling2D, Conv2D

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model

from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.layers import Dense, Concatenate, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
import keras_metrics as km
from sklearn.metrics import f1_score, precision_score, recall_score

from util import get_data_newfour

AUTOTUNE = tf.data.experimental.AUTOTUNE
def channel_attention(h1, ratio=16, name=""):
    hs = GlobalAveragePooling2D()(h1)
    hs = Reshape((1, 1, hs.shape[1]))(hs)
    hs = Conv2D(2048 // 16, kernel_size=1, strides=1, padding="same", kernel_regularizer=regularizers.l2(1e-4),
                use_bias=True, activation="relu")(hs)
    hs = Conv2D(2048, kernel_size=1, strides=1,
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4),
                use_bias=True)(hs)
    # 全局最大
    hb = GlobalMaxPooling2D()(h1)
    hb = Reshape((1, 1, hb.shape[1]))(hb)
    hb = Conv2D(2048 // 16, kernel_size=1, strides=1, padding="same", kernel_regularizer=regularizers.l2(1e-4),
                use_bias=True, activation="relu")(hb)
    hb = Conv2D(2048, kernel_size=1, strides=1, padding="same", kernel_regularizer=regularizers.l2(1e-4),
                use_bias=True)(hb)
    out = hs + hb  # 最大加平均
    out = tf.nn.sigmoid(out)
	
    return out* h1


def spatial_attention(input_feature, name=""):
    kernel_size = 7
    cbam_feature = input_feature


    avg_pool = tf.reduce_mean(cbam_feature, axis=3, keepdims=True)
    max_pool = tf.reduce_max(cbam_feature, axis=3, keepdims=True)

    concat = Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          name="spatial_attention_" + str(name))(concat)
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return input_feature  * cbam_feature

def cbam_block(cbam_feature, ratio=16, name=""):
	cbam_feature = channel_attention(cbam_feature, ratio, name=name)
	cbam_feature = spatial_attention(cbam_feature, name=name)
	return cbam_feature

if __name__ == '__main__':

    train_ds, val_ds = get_data_newfour()

    input_layer = Input(shape=(300, 300, 3))
    resNet = ResNet152(include_top=False, weights=None, input_tensor=input_layer,
                       input_shape=(300, 300, 3))
    xception = Xception(include_top=False, weights=None, input_tensor=input_layer,
                        input_shape=(300, 300, 3))
    top1_model = MaxPooling2D(input_shape=(7, 7, 1024), data_format='channels_last')(resNet.output)
    top2_model = MaxPooling2D(input_shape=(7, 7, 1024), data_format='channels_last')(xception.output)
    concatenate_model = Concatenate(axis=1)([top1_model, top2_model])

    concatenate_model.trainable = False
    out = cbam_block(concatenate_model)
    out = Flatten()(out)
    top_model = Dense(units=512, activation="relu", kernel_regularizer='l2')(out)
    top_model = Dense(units=256, activation="relu", kernel_regularizer='l2')(top_model)
    top_model = Dropout(rate=0.5)(top_model)
    top_model = Dense(units=2, activation="softmax")(top_model)
    model = Model(inputs=input_layer, outputs=top_model)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', km.f1_score(), km.recall(), km.precision()])

    early_stopping = EarlyStopping(

        monitor='val_accuracy',
        verbose=1,
        patience=40,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(min_lr=0.00001,
                                                     factor=0.2)
    # num_0 = len(os.listdir('train_data_all/0'))
    # num_1 = len(os.listdir('train_data_all/1'))
    # num_2 = len(os.listdir('train_data_all/2'))
    # num_3 = len(os.listdir('train_data_all/3'))
    # total = num_0 + num_1+num_3+num_2
    # weight_for_0 = total / num_0 / 4.0
    # weight_for_1 = total / num_1 / 4.0
    # weight_for_2 = total / num_2 / 4.0
    # weight_for_3 = total / num_3 / 4.0
    #
    # class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3}
    # print(class_weight)
    # 迭代次数2000，准确率还可以，耐心等待
    history = model.fit(train_ds,steps_per_epoch=len(train_ds), epochs=1000, callbacks=[early_stopping, reduce_lr], validation_data=val_ds, validation_steps=len(val_ds))


    predict = []

    val_targ = []
    for i in range(6):
        res = model.predict(val_ds[i][0])
        [predict.append(np.argmax(r)) for r in res]
        [val_targ.append(np.argmax(label)) for label in val_ds[i][1]]

    print(len(predict))
    print(len(val_targ))
    print(predict)
    print(val_targ)

    _val_f1 = f1_score(val_targ, predict, average='micro')
    _val_recall = recall_score(val_targ, predict, average=None)
    _val_precision = precision_score(val_targ, predict, average=None)
    print('_val_f1', _val_f1)
    print('_val_recall', _val_recall[0])
    print('_val_precision', _val_precision[0])

    hist_df = pd.DataFrame(history.history)

    y_pre_file = 'cbam-resnet152-xception_predict.csv'
    y_rel_file = 'cbam-resnet152-xception_reltag.csv'
    test1 = pd.DataFrame(data=predict)
    test2 = pd.DataFrame(data=val_targ)
    test1.to_csv(y_pre_file, encoding='utf-8')
    test2.to_csv(y_rel_file, encoding='utf-8')

    hist_csv_file = 'cbam-resnet152-xception_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    # model.save('end_model.h5')