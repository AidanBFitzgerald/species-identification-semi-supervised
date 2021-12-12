from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from tensorflow.keras.utils import Sequence
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
from matplotlib import pyplot as plt
import I3D

LABEL_PERCENT = 0.5


def load(name):
    X = np.load("D:\Coding\Work\species-identification-thermal-imaging\preprocessed" + name + ".npy")
    y = np.load("D:\Coding\Work\species-identification-thermal-imaging\preprocessed" + name + "-labels.npy")
    y_one_hot_encoded = np.zeros([y.shape[0], np.unique(y).size])
    y_one_hot_encoded[range(y.shape[0]), y] = 1
    return X, y_one_hot_encoded


def load_temporal_gradients(name):
    X = np.load("D:\Coding\Work\species-identification-thermal-imaging\preprocessed" + name + ".npy")
    return X


def split_labeled_unlabeled(X_train, y_train, ):
    X_train_unlabeled, X_train_labeled, y_train_unlabeled, y_train_labeled = train_test_split(
        X_train,
        y_train,
        test_size=LABEL_PERCENT,
        random_state=123,
        stratify=y_train
    )
    return X_train_unlabeled, X_train_labeled, y_train_unlabeled, y_train_labeled


class DataGenerator(Sequence):
    def __init__(self, vids, mvm, labels, batch_size, flip=False, angle=0, shuffle=True):
        self.vids = vids
        self.mvm = mvm
        self.labels = labels
        self.indices = np.arange(vids.shape[0])
        self.batch_size = batch_size
        self.flip = flip
        self.angle = angle
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def random_rotate(self, batch, x, y):
        rad = np.random.uniform(-self.angle, self.angle) / 180 * np.pi
        rotm = np.array([[np.cos(rad), np.sin(rad)],
                         [-np.sin(rad), np.cos(rad)]])
        xm, ym = x.mean(), y.mean()
        x, y = np.einsum('ji, mni -> jmn', rotm, np.dstack([x - xm, y - ym]))
        return x + xm, y + ym

    def horizontal_flip(self, batch):
        return np.flip(batch, 3)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        vids = np.array(self.vids[indices])
        x, y = np.meshgrid(np.arange(32) * 0.75, np.arange(32) * 0.75)
        if self.angle:
            x, y = self.random_rotate(vids, x, y)
        if self.flip and np.random.random() < 0.5:
            vids = self.horizontal_flip(vids)
        x = np.clip(x, 0, vids.shape[2] - 1).astype(np.int)
        y = np.clip(y, 0, vids.shape[3] - 1).astype(np.int)
        vids = vids[:, :, x, y].transpose(0, 1, 3, 2, 4)
        if self.mvm is not None:
            out = [vids, self.mvm[indices]], self.labels[indices]
        elif self.labels is not None:
            out = vids, self.labels[indices]
        else:
            out = vids
        return out

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def joint_model():
    MLP = Sequential()
    MLP.add(Flatten())
    MLP.add(Dropout(0.5))
    MLP.add(Dense(256, activation="relu"))
    MLP.add(Dense(13, activation="softmax"))

    inputs = Input((45, 32, 32, 3))
    x = I3D.Inception_Inflated3d(include_top=False,
                                 weights='rgb_imagenet_and_kinetics',
                                 input_shape=(45, 32, 32, 3))(inputs)
    outputs = MLP(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def main():
    print("Dataset loading..", end=" ")
    # Loading the preprocessed videos
    X_train, y_train = load("/training")
    X_val, y_val = load("/validation")
    X_test, y_test = load("/test")
    # Since Keras likes the channels last data format
    X_train = X_train.transpose(0, 1, 3, 4, 2)
    X_val = X_val.transpose(0, 1, 3, 4, 2)
    X_test = X_test.transpose(0, 1, 3, 4, 2)
    print("Dataset loaded!")

    # Split training set into labelled and unlabeled
    X_train_unlabeled, X_train_labeled, y_train_unlabeled, y_train_labeled = split_labeled_unlabeled(X_train, y_train)

    epochs = 100
    batch_size = 32
    learning_rate = 0.001

    model = joint_model()

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
    print(model.summary())

    train_data = DataGenerator(X_train_labeled, None, y_train_labeled, batch_size)
    val_data = DataGenerator(X_val, None, y_val, batch_size)
    test_data = DataGenerator(X_test, None, y_test, batch_size)

    # create log dir
    if not os.path.exists("./logs/I3D-supervised"):
        os.makedirs("./logs/I3D-supervised")

    current_time = str(int(time.time()))

    # csv logs based on the time
    csv_logger = CSVLogger('./logs/I3D-supervised/log_' + current_time + str(LABEL_PERCENT) + '.csv', append=True,
                           separator=';')

    # settings for reducing the learning rate
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

    # save the model at the best epoch
    checkpointer = ModelCheckpoint(filepath='./logs/I3D-supervised/best_model_' + current_time + str(LABEL_PERCENT) + '.hdf5', verbose=1,
                                   save_best_only=True, monitor='val_accuracy', mode='max')

    # Training the model on the training set, with early stopping using the validation set
    callbacks = [EarlyStopping(patience=10), reduce_lr, csv_logger, checkpointer]

    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=val_data,
                        callbacks=callbacks)

    # plot training history
    # two plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    fig.patch.set_facecolor('white')

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'val'], loc='upper left')

    fig.savefig('./logs/I3D-supervised/plot' + current_time + '.svg', format='svg')

    model.load_weights('./logs/I3D-supervised/best_model_' + current_time + str(LABEL_PERCENT) + '.hdf5')

    # evalutate accuracy on hold out set
    eval_metrics = model.evaluate(test_data, verbose=0)
    for idx, metric in enumerate(model.metrics_names):
        if metric == 'accuracy':
            print(metric + ' on hold out set:', round(100 * eval_metrics[idx], 1), "%", sep="")

    # Evaluating the final model on the test set
    y_pred = np.argmax(model.predict(test_data), axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_test_size = (y_test.shape[0] // batch_size) * batch_size
    y_test = y_test[y_test_size]
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
