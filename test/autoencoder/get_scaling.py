from sklearn.externals import joblib
from keras.models import load_model
import numpy as np

scaler = joblib.load("scaler.bin")
encoder = load_model("encoder.h5")
data = np.load("data.npy")

data_scl = scaler.transform(data)
data_enc = encoder.predict(data_scl)
print(data_enc.shape)
print("Max: {}".format(np.max(data_enc, axis=0)))
print("Min: {}".format(np.min(data_enc, axis=0)))
