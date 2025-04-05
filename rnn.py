import numpy as np
from tensorflow.python.layers.core import Dropout

from const import *
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  LSTM, Dense

def fix_hand_array(arr):
    base_len = 132
    hand_len = 21 * 3  # 63
    expected_len = base_len + hand_len * 2  # 258

    # If array is already correct
    if len(arr) == expected_len:
        return arr

    # Find how many extra zeros were added
    extra = len(arr) - expected_len

    if extra not in [21, 42]:  # only allow 1 or 2 missing hands
        raise ValueError(f"Unexpected array length: {len(arr)}")

    # Copy the valid part
    fixed = list(arr[:base_len])

    # This part might contain extra padding
    remainder = arr[base_len:]

    i = 0
    for hand in ['left', 'right']:
        hand_data = remainder[i:i + hand_len]
        i += hand_len

        # Check if this is all zeros or not
        if np.allclose(hand_data, 0):
            # Remove extra 21 zeros
            i += 21  # skip the extra padding
            fixed.extend(np.zeros(hand_len))  # add correct padding
        else:
            fixed.extend(hand_data)

    return np.array(fixed)



def fix_data():
    for symbol in SYMBOLS:
        for vid in range(VID_NUM):
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
        for vid in range(VID_NUM):
            #print(f"Video {vid}")
            window = []
            for fr in range(FRAMES):
                #print(f"Frame {fr}")
                res = np.load(os.path.join(DATA_DIR,symbol,str(vid),f"{fr}.npy"))
                window.append(res)
            sequences.append(np.array(window))
            labels.append(SYMBOL_MAPS[symbol])
    y = to_categorical(labels).astype(int)
    return np.array(sequences),y

def train_model(x,y):
    x=x.reshape((1800,15,258))
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.1)
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, activation='relu', input_shape=(15, 258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(SYMBOLS.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(train_x, train_y,epochs=100,batch_size=256)

    loss, accuracy = model.evaluate(test_x,test_y)
    print(accuracy)
    if input("Save Model? (Y/N): ").lower() == 'y':
        model.save('model.keras')

def validate():
    for symbol in SYMBOLS[26:]:
        assert len(os.listdir(os.path.join(DATA_DIR,symbol)))==30

#fix_data()

x,y = load_data()
print(x.shape)
print(y.shape)
train_model(x,y)