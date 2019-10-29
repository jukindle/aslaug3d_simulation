from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np


encoding_dim = 16

data = np.load("data.npy")

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

input = Input(shape=(data.shape[1],))

l1 = Dense(64, activation='relu')(input)
l2 = Dense(64, activation='relu')(l1)
l3 = Dense(32, activation='relu')(l2)
encoded = Dense(encoding_dim, activation='relu')(l3)
l4 = Dense(32, activation='relu')(encoded)
l5 = Dense(64, activation='relu')(l4)
l6 = Dense(64, activation='relu')(l5)
decoded = Dense(data.shape[1], activation='relu')(l6)

encoder = Model(input, encoded)
autoencoder = Model(input, decoded)

autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(data, data, epochs=500, batch_size=256, shuffle=True)

joblib.dump(scaler, "scaler.bin")
encoder.save("encoder.h5")
