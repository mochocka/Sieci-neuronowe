0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import Callback
from openpyxl import Workbook
import tensorflow as tf


def relu(x):
    return tf.maximum(0.0, x)

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def tanh(x):
    return tf.tanh(x)

def softmax(x):
    e_x = tf.exp(x - tf.reduce_max(x, axis=-1, keepdims=True))
    return e_x / tf.reduce_sum(e_x, axis=-1, keepdims=True)


def accuracy(y, output):
    y_binary = (y > 0.5).astype(int)
    output_binary = (output > 0.5).astype(int)
    return np.mean(y_binary == output_binary)


def prepare_data(file_path):
    data = pd.read_excel(file_path, header=0)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()

    features = data.columns[:-1].tolist()
    target = data.columns[-1]

    X = data[features]
    y = data[target]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
    )

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y



def model_builder(X_train, num_layers, num_neurons, activation_func, optimizer):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  

    for _ in range(num_layers):
        model.add(Dense(num_neurons))
        model.add(Lambda(activation_func))  

    model.add(Dense(1))
    model.add(Lambda(sigmoid))  

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae'])
    return model


class TrainingLoggerCallback(Callback):
    def __init__(self, X_train, y_train, X_test, y_test, scaler_y, wb_train_epoch, wb_test_epoch):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scaler_y = scaler_y
        self.wb_train_epoch = wb_train_epoch
        self.wb_test_epoch = wb_test_epoch
        self.history_train = []
        self.history_test = []

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train)
        y_train_original = self.scaler_y.inverse_transform(self.y_train)
        acc_train = accuracy(y_train_original, y_train_pred)

        self.wb_train_epoch.append([epoch + 1, acc_train, logs['loss'], logs['mse'], logs['mae']])

        y_test_pred = self.model.predict(self.X_test)
        acc_test = logs['val_accuracy']  

        self.wb_test_epoch.append([epoch + 1, acc_test, logs['val_loss'], logs['val_mse'], logs['val_mae']])

        self.history_train.append(logs)
        self.history_test.append({
            'val_loss': logs['val_loss'],
            'val_accuracy': logs['val_accuracy'],
            'val_mse': logs['val_mse'],
            'val_mae': logs['val_mae']
        })


def generate_plots(history_train, history_test):
    epochs = range(1, len(history_train) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [log['accuracy'] for log in history_train], label='Train', color='red')
    plt.plot(epochs, [log['val_accuracy'] for log in history_test], label='Test', color='blue')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [log['loss'] for log in history_train], label='Train', color='red')
    plt.plot(epochs, [log['val_loss'] for log in history_test], label='Test', color='blue')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()



def run_experiments():
    layers_options = [1]
    neurons_options = [4]
    activation_options = [relu, tanh, sigmoid, softmax] 
    optimizer_options = ['adam']

    wb_train_epoch = Workbook()
    ws_train_epoch = wb_train_epoch.active
    ws_train_epoch.append(['Epoch', 'Accuracy', 'Loss', 'MSE', 'MAE'])

    wb_test_epoch = Workbook()
    ws_test_epoch = wb_test_epoch.active
    ws_test_epoch.append(['Epoch', 'Accuracy', 'Loss', 'MSE', 'MAE'])

    file_path = 'dane.xlsx'
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_data(file_path)

    for layers in layers_options:
        for neurons in neurons_options:
            for activation_func in activation_options:
                for optimizer in optimizer_options:
                    print(f"Training: layers={layers}, neurons={neurons}, activation={activation_func.__name__}, optimizer={optimizer}")

                    model = model_builder(X_train, layers, neurons, activation_func, optimizer)
                    training_logger = TrainingLoggerCallback(X_train, y_train, X_test, y_test, scaler_y, ws_train_epoch, ws_test_epoch)
                    model.fit(
                        X_train, y_train,
                        epochs=100, batch_size=32, verbose=0, validation_split=0.2, callbacks=[training_logger]
                    )

                    wb_train_epoch.save('train_epoch_results.xlsx')
                    wb_test_epoch.save('test_epoch_results.xlsx')

                    generate_plots(training_logger.history_train, training_logger.history_test)


run_experiments()
