import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy
from tensorflow.keras.layers import Lambda


class AslaugPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, **kwargs):
        super(AslaugPolicy, self).__init__(sess, ob_space, ac_space,
                                           n_env, n_steps, n_batch,
                                           reuse=reuse, scale=False)

        # Scale observation
        with tf.variable_scope("observation_scaling", reuse=reuse):
            obs_avg = tf.constant((self.ob_space.high+self.ob_space.low)/2.0,
                                  name="obs_avg")
            obs_dif = tf.constant((self.ob_space.high - self.ob_space.low),
                                  name="obs_diff")
            shifted = tf.math.subtract(self.processed_obs, obs_avg)
            proc_obs = tf.math.divide(shifted, obs_dif)

        lrelu = tf.nn.leaky_relu

        with tf.variable_scope("model/main_block", reuse=reuse):
            l1 = tf.layers.Dense(64, activation=lrelu)(proc_obs)
            l2 = tf.layers.Dense(64, activation=lrelu)(l1)
            l3 = tf.layers.Dense(32, activation=lrelu)(l2)
            l4 = tf.layers.Dense(32, activation=lrelu)(l3)
            c_out = tf.layers.Dense(16, activation=lrelu)(l4)

        with tf.variable_scope("model/actor_critic_block", reuse=reuse):
            vf_latent = tf.layers.Dense(32, activation=lrelu,
                                        name="vf_latent")(c_out)
            pi_latent = tf.layers.Dense(16, activation=lrelu,
                                        name="vf_f_1")(c_out)

            value_fn = tf.layers.Dense(1, name='vf')(vf_latent)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent,
                                                           vf_latent,
                                                           init_scale=1.0)

        self._value_fn = value_fn
        # self._initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action,
                                                    self.value_flat,
                                                    self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action,
                                                    self.value_flat,
                                                    self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def crop(self, dimension, start, end):
        # Crops (or slices) a Tensor on a given dimension from start to end
        def func(x):
            if dimension == 0:
                return x[start: end]
            if dimension == 1:
                return x[:, start: end]
            if dimension == 2:
                return x[:, :, start: end]
            if dimension == 3:
                return x[:, :, :, start: end]
            if dimension == 4:
                return x[:, :, :, :, start: end]
        return Lambda(func)
