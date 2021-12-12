
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

LABEL_PERCENT = 0.1


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
    def __init__(self, vids, mvm, labels, batch_size, flip=False, angle=0, noise=False, crop=0, shift=0, shuffle=True):
        self.vids = vids
        self.mvm = mvm
        self.labels = labels
        self.indices = np.arange(vids.shape[0])
        self.batch_size = batch_size
        self.flip = flip
        self.shift = shift
        self.angle = angle
        self.shuffle = shuffle
        self.on_epoch_end()
        self.crop = crop
        self.noise = noise

    def __len__(self):
        return len(self.indices) // self.batch_size

    def random_rotate(self, batch, x, y):
        rad = np.random.uniform(-self.angle, self.angle) / 180 * np.pi
        rotm = np.array([[np.cos(rad), np.sin(rad)],
                         [-np.sin(rad), np.cos(rad)]])
        xm, ym = x.mean(), y.mean()
        x, y = np.einsum('ji, mni -> jmn', rotm, np.dstack([x - xm, y - ym]))
        return x + xm, y + ym

    def random_translate(self, batch, x, y):
        xs = np.random.uniform(-self.shift, self.shift)
        ys = np.random.uniform(-self.shift, self.shift)
        return x + xs, y + ys

    def horizontal_flip(self, batch):
        batch = batch.copy()
        for i in range(len(batch)):
            if np.random.random() > 0.5:
                batch[i] = np.flip(batch[i], 3)
        return batch

    def random_zoom(self, batch, x, y):
        ax = np.random.uniform(self.crop)
        bx = np.random.uniform(ax)
        ay = np.random.uniform(self.crop)
        by = np.random.uniform(ay)
        x = x * (1 - ax / batch.shape[2]) + bx
        y = y * (1 - ay / batch.shape[3]) + by
        return x, y

    def gaussian_noise(self,vids, mean=0, sigma=0.03):
        vids = vids.copy()
        for i in range(len(vids)):
            if np.random.random() > 0.5:
                noise = np.random.normal(mean, sigma, vids[i].shape)
                mask_overflow_upper = vids[i] + noise >= 1.0
                mask_overflow_lower = vids[i] + noise < 0
                noise[mask_overflow_upper] = 1.0
                noise[mask_overflow_lower] = 0
                vids[i] += noise
        return vids

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        vids = np.array(self.vids[indices])
        # x, y = np.meshgrid(range(vids.shape[2]), range(vids.shape[3]))
        x, y = np.meshgrid(np.arange(32) * 0.75, np.arange(32) * 0.75)
        if self.angle:
            x, y = self.random_rotate(vids, x, y)

        if self.shift:
            x, y = self.random_translate(vids, x, y)

        if self.flip:
            vids = self.horizontal_flip(vids)

        if self.noise:
            vids = self.gaussian_noise(vids)

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
    X_temporal_gradient_train = load_temporal_gradients("/training-temporal-gradients")
    # Since Keras likes the channels last data format
    X_train = X_train.transpose(0, 1, 3, 4, 2)
    X_val = X_val.transpose(0, 1, 3, 4, 2)
    X_test = X_test.transpose(0, 1, 3, 4, 2)
    X_temporal_gradient_train = X_temporal_gradient_train.transpose(0, 1, 3, 4, 2)
    print("Dataset loaded!")

    # Split training set into labelled and unlabeled
    X_train_unlabeled, X_train_labeled, y_train_unlabeled, y_train_labeled = split_labeled_unlabeled(X_train, y_train)
    X_train_unlabeled_temporal, X_train_labeled_temporal, y_train_unlabeled_temporal, y_train_labeled_temporal = \
        split_labeled_unlabeled(X_temporal_gradient_train, y_train)

    print(y_train_unlabeled_temporal[:10])
    print(y_train_unlabeled[:10])

    epochs = 100
    batch_size = 32
    unlabelled_batch_size = batch_size * 7
    learning_rate = 0.001
    threshold = 0.9

    model = joint_model()

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
    print(model.summary())
    # combine the labeled thermal and temporal data
    X_train_combined = np.append(X_train_labeled, X_train_labeled_temporal, axis=0)
    y_train_combined = np.append(y_train_labeled, y_train_labeled_temporal, axis=0)
    train_data = DataGenerator(X_train_combined, None, y_train_combined, batch_size)
    pseudo_data_weak_aug = DataGenerator(X_train_unlabeled, None, None, unlabelled_batch_size, True, shuffle=False)
    pseudo_data_weak_aug_temporal = DataGenerator(X_train_unlabeled_temporal, None, None, unlabelled_batch_size, True, shuffle=False)
    val_data = DataGenerator(X_val, None, y_val, batch_size)
    test_data = DataGenerator(X_test, None, y_test, batch_size)

    # create log dir
    if not os.path.exists("./logs/I3D"):
        os.makedirs("./logs/I3D")

    current_time = str(int(time.time()))

    # csv logs based on the time
    csv_logger = CSVLogger('./logs/I3D/log_' + current_time + 'temporal.csv', append=True, separator=';')

    # settings for reducing the learning rate
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

    # save the model at the best epoch
    checkpointer = ModelCheckpoint(filepath='./logs/I3D/best_model_' + current_time + 'temporal.hdf5', verbose=1,
                                   save_best_only=True, monitor='val_accuracy', mode='max')

    # Training the model on the training set, with early stopping using the validation set
    callbacks = [EarlyStopping(patience=10), reduce_lr, csv_logger, checkpointer]

    history = model.fit(train_data,
                        epochs=5,
                        validation_data=val_data,
                        callbacks=callbacks)


    for epoch_idx in range(epochs):
        print("Epoch {}/{}".format(epoch_idx + 1, epochs))
        pseudo_labels = model.predict(pseudo_data_weak_aug)
        pseudo_labels_temporal = model.predict(pseudo_data_weak_aug_temporal)
        # average prediction from views
        pseudo_labels = (pseudo_labels_temporal + pseudo_labels)/2
        # Only accept labels with certain threshold accuracy in their probability
        pseudo_labels = np.array([np.argmax(x) if np.max(x) > threshold else -1 for x in pseudo_labels])
        pseudo_dataset = np.array([[x, y] for x, y in zip(X_train_unlabeled, pseudo_labels) if y != -1])
        if len(pseudo_dataset) > 0:
            pseudo_dataset = np.append(np.array([[x, y] for x, y in zip(X_train_unlabeled_temporal, pseudo_labels) if y != -1]),
                                   pseudo_dataset, axis=0)
        pseudo_dataset = np.array(pseudo_dataset)
        x_values = np.array([x for x, y in pseudo_dataset])
        y_values = np.array([y for x, y in pseudo_dataset])
        y_pseudo = np.zeros((y_values.shape[0], 13))
        for i in range(y_values.shape[0]):
            y_pseudo[i][y_values[i]] = 1
        if len(y_values) >= unlabelled_batch_size:
            pseudo_data_strong_aug = DataGenerator(x_values, None, y_pseudo, unlabelled_batch_size, True, 10, True)
            model.fit(pseudo_data_strong_aug, epochs=1, validation_data=val_data, callbacks=callbacks)

        model.fit(train_data, epochs=1, validation_data=val_data, callbacks=callbacks)

        #learning_rate = learning_rate * 0.5 * (math.cos(((epoch_idx + 1)/epochs) * math.pi) + 1)

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

    fig.savefig('./logs/I3D/plot' + current_time + '.svg', format='svg')

    model.load_weights('./logs/I3D/best_model_' + current_time + 'temporal.hdf5')

    # evalutate accuracy on hold out set
    eval_metrics = model.evaluate(test_data, verbose=0)
    for idx, metric in enumerate(model.metrics_names):
        if metric == 'accuracy':
            print(metric + ' on hold out set:', round(100 * eval_metrics[idx], 1), "%", sep="")

    # Evaluating the final model on the test set
    # y_pred = np.argmax(model.predict(test_data), axis=1)
    # y_test = np.argmax(y_test, axis=1)
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
