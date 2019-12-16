from multiprocessing.connection import Listener
import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, \
                         Concatenate, Lambda
from keras.models import Model


class Discriminator:
    def __init__(self, params, obs_slicing, obs_space):
        self.params = params
        self.obs_slicing = obs_slicing
        self.obs_space = obs_space

        address = ('localhost', params["port"])
        self.listener = Listener(address, authkey=b"tuemecrifuieR11NAF10")

    def main(self):
        self.discr_model = DiscriminatorModel(self.params, self.obs_slicing,
                                              self.obs_space)
        while True:
            try:
                conn = self.listener.accept()
                msg = conn.recv()
                # print(msg["cmd"])
                if msg["cmd"] == "add":
                    x, y = msg["data"]
                    self.discr_model.add_training_data(x, y)
                    resp = {"type": "info", "data": "ok"}
                    conn.send(resp)
                elif msg["cmd"] == "predict":
                    x, = msg["data"]
                    y = self.discr_model.predict(x)
                    resp = {"type": "prediction", "data": y}
                    conn.send(resp)

            except Exception as e:
                print(e)
                pass


class DiscriminatorModel:
    def __init__(self, params, obs_slicing, obs_space):
        self.params = params
        self.buffer_x = []
        self.buffer_y = []

        # Create and compile model
        self.model = self.create_model(obs_slicing)
        self.model.compile(optimizer='adam', loss='mse')
        tb_cb = keras.callbacks.TensorBoard(log_dir='data/tb_logs/discrim')
        self.callbacks = [tb_cb]

        # Prepare scalings and slicings
        obs_avg = (obs_space.high + obs_space.low)/2.0
        obs_dif = (obs_space.high - obs_space.low)
        self.scale_input = lambda x: (x - obs_avg)/obs_dif
        self.split_input = lambda x: np.split(x, obs_slicing[1:-1], axis=1)
        rew_min, rew_max = params["reward_scaling"]
        rew_avg = (rew_min + rew_max)/2.0
        rew_dif = (rew_max - rew_min)
        self.scale_rew = lambda x: np.clip((x - rew_avg)/rew_dif, -1, 1)
        self.unscale_rew = lambda x: x * rew_dif + rew_avg

    def predict(self, x):
        x_1 = self.scale_input(x)
        x_2 = self.split_input(x_1)
        pred = self.model.predict(x_2)
        return self.unscale_rew(pred)

    def add_training_data(self, x, y):
        self.buffer_x.append(self.scale_input(x))
        self.buffer_y.append(self.scale_rew(y))
        if len(self.buffer_x) >= self.params["buffer_length"]:
            print("Training discriminator")
            x = self.split_input(np.array(self.buffer_x))
            y = np.array(self.buffer_y)
            self.model.fit(x, y, batch_size=self.params["batch_size"],
                           epochs=self.params["epochs"],
                           callbacks=self.callbacks)
            self.buffer_x = []
            self.buffer_y = []

    def create_model(self, obs_slicing):
        o = obs_slicing
        in_sp = Input(shape=(o[1] - o[0],))
        in_mb = Input(shape=(o[2] - o[1],))
        in_lp = Input(shape=(o[3] - o[2],))
        in_jp = Input(shape=(o[4] - o[3],))
        in_jv = Input(shape=(o[5] - o[4],))
        in_sc1 = Input(shape=(o[6] - o[5],))
        in_sc2 = Input(shape=(o[7] - o[6],))

        s1_0 = Lambda(lambda x: keras.backend.expand_dims(x))(in_sc1)
        s1_1 = Conv1D(4, 11, activation='relu', name="s1_1")(s1_0)
        s1_2 = Conv1D(8, 7, activation='relu', name="s1_2")(s1_1)
        s1_3 = MaxPooling1D(10, 8, name="s1_3")(s1_2)
        s1_4 = Conv1D(8, 7, activation='relu', name="s1_4")(s1_3)
        s1_5 = Conv1D(4, 5, activation='relu', name="s1_5")(s1_4)
        s1_6 = Flatten(name="s1_6")(s1_5)
        s1_7 = Dense(64, activation='relu', name="s1_7")(s1_6)
        s1_out = Dense(32, activation='relu', name="s1_out")(s1_7)

        s2_0 = Lambda(lambda x: keras.backend.expand_dims(x))(in_sc2)
        s2_1 = Conv1D(4, 11, activation='relu', name="s2_1")(s2_0)
        s2_2 = Conv1D(8, 7, activation='relu', name="s2_2")(s2_1)
        s2_3 = MaxPooling1D(10, 8, name="s2_3")(s2_2)
        s2_4 = Conv1D(8, 7, activation='relu', name="s2_4")(s2_3)
        s2_5 = Conv1D(4, 5, activation='relu', name="s2_5")(s2_4)
        s2_6 = Flatten(name="s2_6")(s2_5)
        s2_7 = Dense(64, activation='relu', name="s2_7")(s2_6)
        s2_out = Dense(32, activation='relu', name="s2_out")(s2_7)

        sc_0 = Concatenate()([s1_out, s2_out])
        sc_1 = Dense(32, activation='relu')(sc_0)
        sc_2 = Dense(16, activation='relu')(sc_1)

        t_0 = Concatenate()([sc_2, in_sp, in_mb, in_lp, in_jp, in_jv])
        t_1 = Dense(128, activation='relu')(t_0)
        t_2 = Dense(64, activation='relu')(t_1)
        t_3 = Dense(32, activation='relu')(t_2)
        t_4 = Dense(16, activation='relu')(t_3)
        out = Dense(1, activation='tanh')(t_4)

        inputs = [in_sp, in_mb, in_lp, in_jp, in_jv, in_sc1, in_sc2]
        outputs = [out]
        return Model(inputs=inputs, outputs=outputs)
