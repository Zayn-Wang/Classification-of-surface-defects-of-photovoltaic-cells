import os
from tensorflow.python.keras import regularizers, Input, Model
from tensorflow.keras.applications import ResNet152, Xception
from tensorflow.python.keras.api.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.layers import InputSpec, Concatenate, Dense, Dropout, GlobalAveragePooling2D, Reshape, Conv2D, \
    Flatten, GlobalMaxPooling2D
from tensorflow.keras import backend as K, Input, Model
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Concatenate, Dropout
from tensorflow.python.keras.layers import BatchNormalization
# from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Rescaling
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Concatenate, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
import keras_metrics as km
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.densenet import DenseNet169, DenseNet121
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet import ResNet50, ResNet152
from tensorflow.python.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Flatten, GlobalMaxPooling2D, \
    MaxPooling2D, Conv2D
from tensorflow.python.keras import regularizers, Input, Model

import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.applications.resnet import ResNet152

from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Concatenate, Dropout
from tensorflow.python.keras.layers import BatchNormalization
# from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Rescaling
from util import get_data_newfour
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Concatenate, Dropout, AveragePooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':
    train_ds, val_ds = get_data_newfour()
    input_layer = Input(shape = (300, 300, 3))
    resNet = ResNet152(include_top=False, input_tensor=input_layer,
                       input_shape=(300, 300, 3))
    xception = Xception(include_top=False, input_tensor=input_layer,
                        input_shape=(300, 300, 3))
    top1_model = MaxPooling2D(data_format = 'channels_last')(resNet.output)
    top2_model = MaxPooling2D(data_format = 'channels_last')(xception.output)
    concatenate_model = Concatenate(axis = 1)([top1_model, top2_model])
    concatenate_model.trainable = False

    h1 = BatchNormalization()(concatenate_model)

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
    # hb = GlobalAveragePooling2D()(h1)
    hb = Reshape((1, 1, hb.shape[1]))(hb)
    hb = Conv2D(2048 // 16, kernel_size=1, strides=1, padding="same", kernel_regularizer=regularizers.l2(1e-4),
                use_bias=True, activation="relu")(hb)
    hb = Conv2D(2048, kernel_size=1, strides=1, padding="same", kernel_regularizer=regularizers.l2(1e-4),
                use_bias=True)(hb)
    out = hs + hb  # 最大加平均
    out = tf.nn.sigmoid(out)
    out = out * h1
    out = Flatten()(out)
    top_model = Dense(units = 512, activation = "relu")(out)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(units = 256, activation = "relu")(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(units = 2, activation = "softmax")(top_model)
    model = Model(inputs = input_layer, outputs = top_model)

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy',km.f1_score(),km.recall(),km.precision()])

    early_stopping = EarlyStopping(

        monitor = 'val_accuracy',
        verbose = 1,
        patience = 20,
        restore_best_weights = True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(min_lr=0.00001,
                                                     factor=0.2)
    num_0 = len(os.listdir('test_data_all/0'))
    num_1 = len(os.listdir('test_data_all/1'))
    total = num_0 + num_1
    # num_2 = len(os.listdir('test_data_all/2'))
    # num_3 = len(os.listdir('test_data_all/1'))
    # total = num_0 + num_1+num_3+num_2
    weight_for_0 = total / num_0 / 2.0
    weight_for_1 = total / num_1 / 2.0
    # weight_for_2 = total / num_2 / 4.0
    # weight_for_3 = total / num_3 / 4.0
    #
    # class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 1: weight_for_3}
    class_weight = {0: weight_for_0, 1: weight_for_1}
    # print(class_weight)
    # 迭代次数2000，准确率还可以，耐心等待
    history = model.fit(train_ds,steps_per_epoch=len(train_ds), epochs=1000, callbacks=[early_stopping, reduce_lr], validation_data=val_ds, validation_steps=len(val_ds), class_weight=class_weight)

    predict = []

    val_targ =[]
    for i in range(14):
       res = model.predict(val_ds[i][0])
       [predict.append(np.argmax(r)) for r in res]
       [val_targ.append(np.argmax(label)) for label in val_ds[i][1]]

    print(len(predict))
    print(len(val_targ))
    print(predict)
    print(val_targ)

    _val_f1 = f1_score(val_targ, predict,average='micro')
    _val_recall = recall_score(val_targ, predict,average=None)###
    _val_precision = precision_score(val_targ, predict,average=None)###
    print('_val_f1',_val_f1)
    print('_val_recall',_val_recall[0])
    print('_val_precision',_val_precision[0])

    hist_df = pd.DataFrame(history.history)

    y_pre_file = 'ResNet152+Xception_predict.csv'
    y_rel_file = 'ResNet152+Xception_reltag.csv'
    test1=pd.DataFrame(data=predict)
    test2=pd.DataFrame(data=val_targ)
    test1.to_csv(y_pre_file, encoding= 'utf-8')
    test2.to_csv(y_rel_file, encoding= 'utf-8')


    hist_csv_file = 'ResNet152+Xception_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    # model.save('Test_SE_ResNet152+Xception.h5')
