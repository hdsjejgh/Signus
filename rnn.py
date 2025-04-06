import numpy as np
from const import *
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  LSTM, Dense,Dropout, Bidirectional, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt


def fix_hand_array(arr):
    base_len = 132
    hand_len = 21 * 3  # 63
    expected_len = base_len + hand_len * 2
    if len(arr) == expected_len:
        return arr
    extra = len(arr) - expected_len
    fixed = list(arr[:base_len])
    remainder = arr[base_len:]

    i = 0
    for hand in ['left', 'right']:
        hand_data = remainder[i:i + hand_len]
        i += hand_len

        if np.allclose(hand_data, 0):
            i += 21
            fixed.extend(np.zeros(hand_len))
        else:
            fixed.extend(hand_data)
    return np.array(fixed)



def fix_data():
    for symbol in SYMBOLS:
        for vid in range(90):
            if len(os.listdir(os.path.join(DATA_DIR,str(symbol),str(vid)))) == 0:
                continue
            for frame in range(FRAMES):
                nppath = os.path.join(DATA_DIR,symbol,str(vid),str(frame)+".npy")
                array = np.load(nppath)
                array = fix_hand_array(array)
                print(len(array))
                np.save(os.path.join(DATA_DIR,str(symbol),str(vid),str(frame)), array)





def load_data():
    sequences,labels = [],[]
    for symbol in SYMBOLS:
        print(symbol)
        for vid in range(60):
            if len(os.listdir(os.path.join(DATA_DIR,str(symbol),str(vid)))) == 0:
                continue
            #print(f"Video {vid}")
            window = []
            for fr in range(FRAMES):
                #print(f"Frame {fr}")
                res = np.load(os.path.join(DATA_DIR,symbol,str(vid),f"{fr}.npy"))
                if len(res)==258:
                    window.append(res[132:])
                elif len(res)==258-132:
                    window.append(res)
                else:
                    print(len(res))
            sequences.append(np.array(window))
            labels.append(SYMBOL_MAPS[symbol])
    y = to_categorical(labels).astype(int)

    return np.array(sequences),y

def train_model(x,y):
    x=x.reshape((3600,15,126))
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.4)

    early_stop = EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)

    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(15, 126))))
    model.add(Bidirectional(LSTM(128, return_sequences=True,)))
    model.add(Bidirectional(LSTM(256, return_sequences=False,)))
    model.add(LayerNormalization())
    model.add(Dense(128, activation='relu',kernel_regularizer=l2(1e-4)))
    #model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(SYMBOLS.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    history = model.fit(train_x, train_y,validation_data=(test_x,test_y),epochs=40,batch_size=128,callbacks=[early_stop])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['categorical_accuracy'], label='Train Acc')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.show()

    loss, accuracy = model.evaluate(test_x,test_y)
    print(accuracy)
    if input("Save Model? (Y/N): ").lower() == 'y':
        model.save('model.keras')

def validate():
    for symbol in SYMBOLS[26:]:
        assert len(os.listdir(os.path.join(DATA_DIR,str(symbol))))==30

#fix_data()

x,y = load_data()
print(x.shape)
print(y.shape)
train_model(x,y)
