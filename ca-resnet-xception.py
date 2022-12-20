from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Rescaling, BatchNormalization, AvgPool2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Flatten, GlobalMaxPooling2D, \
    MaxPooling2D, Conv2D
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.resnet import ResNet152

from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Concatenate, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
import keras_metrics as km
from sklearn.metrics import f1_score, precision_score, recall_score

from util import get_data_newfour

AUTOTUNE = tf.data.experimental.AUTOTUNE
def CoordAtt(x, reduction = 32):
 
    def coord_act(x):
        tmpx = tf.nn.relu6(x+3) / 6
        x = x * tmpx
        return x
 
    x_shape = x.get_shape().as_list()
    print(x_shape)
    [b, h, w, c] = x_shape
    x_h = AvgPool2D(pool_size=(1, w), strides = 1)(x)
    x_w = AvgPool2D(pool_size=(h, 1), strides = 1)(x)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
 
    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = Conv2D(mip, (1, 1), strides=1, activation=coord_act,name='ca_conv1')(y)
 
    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    a_h = Conv2D(c, (1, 1), strides=1,activation=tf.nn.sigmoid,name='ca_conv2')(x_h)
    a_w = Conv2D(c, (1, 1), strides=1,activation=tf.nn.sigmoid,name='ca_conv3')(x_w)
 
    out = x * a_h * a_w

    return out


if __name__ == '__main__':
    train_ds, val_ds = get_data_newfour()

    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
    # normalization_layer = Rescaling(1. / 255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    input_layer = Input(shape = (300, 300, 3))
    dense = ResNet152(include_top = False , input_tensor = input_layer,
                        input_shape = (300, 300, 3))
    xception = Xception(include_top = False, input_tensor = input_layer,
                        input_shape = (300, 300, 3))
    top1_model = MaxPooling2D(input_shape = (7, 7, 1024), data_format = 'channels_last')(dense.output)
    top2_model = MaxPooling2D(input_shape = (7, 7, 1024), data_format = 'channels_last')(xception.output)
    concatenate_model = Concatenate(axis = 1)([top1_model, top2_model])
    concatenate_model.trainable = False
    # h1 = MaxPooling2D(pool_size = 2)(concatenate_model)
    out  = CoordAtt(concatenate_model)
    out = Flatten()(out)
    top_model = Dense(units = 512, activation = "relu")(out)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(units = 256, activation = "relu")(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(units = 4, activation = "softmax")(top_model)
    model = Model(inputs = input_layer, outputs = top_model)

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
    history = model.fit(train_ds, epochs=1000, callbacks=[early_stopping, reduce_lr], validation_data=val_ds)

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
    _val_recall = recall_score(val_targ, predict, average=None)  ###
    _val_precision = precision_score(val_targ, predict, average=None)  ###
    print('_val_f1', _val_f1)
    print('_val_recall', _val_recall[0])
    print('_val_precision', _val_precision[0])

    hist_df = pd.DataFrame(history.history)

    y_pre_file = 'ca-resnet152-xception_predict.csv'
    y_rel_file = 'ca-resnet152-xception_reltag.csv'
    test1 = pd.DataFrame(data=predict)
    test2 = pd.DataFrame(data=val_targ)
    test1.to_csv(y_pre_file, encoding='utf-8')
    test2.to_csv(y_rel_file, encoding='utf-8')

    hist_csv_file = 'ca-resnet152-xception_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    # model.save('end_model.h5')