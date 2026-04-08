

import numpy as np
import tensorflow as tf

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            normalize=False):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
       Args:
         args: a 2D Tensor or a list of 2D, batch x n, Tensors.
         output_size: int, second dimension of W[i].
         bias: boolean, whether to add a bias term or not.
         bias_initializer: starting value to initialize the bias
           (default is all zeros).
         kernel_initializer: starting value to initialize the weight.
         kernel_regularizer: kernel regularizer
         bias_regularizer: bias regularizer
       Returns:
         A 2D Tensor with shape [batch x output_size] equal to
            sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
       Raises:
         ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    total_arg_size = 0
    shapes = [a.shape for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1] is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "  
                             "but saw %s" % (shape, shape[1]))
        total_arg_size += shape[1]

    dtype = args[0].dtype

    weights = tf.keras.layers.Dense(output_size,
                                     use_bias=bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer)(tf.concat(args, axis=1))

    # Now the computation.
    res = tf.matmul(tf.concat(args, axis=1), weights)

    if normalize:
        res = tf.keras.layers.LayerNormalization()(res)

    if not bias or normalize:
        return res

    if bias_initializer is None:
        bias_initializer = tf.zeros_initializer()
    biases = tf.Variable(
        bias_initializer(shape=[output_size], dtype=dtype),
        name=_BIAS_VARIABLE_NAME,
        trainable=True,
        regularizer=bias_regularizer
    )

    return res + biases


class SimpleRANCell(tf.keras.layers.Layer):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393).

    This is an implementation of the simplified RAN cell described in Equation group (2)."""

    def __init__(self, num_units, input_size=None, normalize=False):
        if input_size is not None:
            print(f"{self}: The input_size parameter is deprecated.")
        self._normalize = normalize
        self._num_units = num_units

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        value = tf.nn.sigmoid(_linear([state, inputs], 2 * self._num_units, True, normalize=self._normalize))
        i, f = tf.split(value=value, num_or_size_splits=2, axis=1)

        new_c = i * inputs + f * state
        return new_c, new_c


class RANStateTuple(tf.keras.layers.Layer):
    pass


class RANCell(tf.keras.layers.Layer):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)

    This is an implementation of the standard RAN cell described in Equation group (1)."""

    def __init__(self, num_units, input_size=None, activation='tanh', normalize=False):
        if input_size is not None:
            print(f"{self}: The input_size parameter is deprecated.")
        self._num_units = num_units
        self._activation = activation if activation is not None else tf.keras.activations.tanh
        self._normalize = normalize

    @property
    def state_size(self):
        return RANStateTuple(self._num_units, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        c, h = state
        gates = tf.nn.sigmoid(_linear([inputs, h], 2 * self._num_units, True,
                                       normalize=self._normalize))

        i, f = tf.split(value=gates, num_or_size_splits=2, axis=1)

        content = _linear([inputs], self._num_units, True,
                          normalize=self._normalize)

        new_c = i * content + f * c

        new_h = self._activation(new_c)

        new_state = RANStateTuple(new_c, new_h)
        return new_h, new_state


class VHRANCell(tf.keras.layers.Layer):
    """Variational Highway variant of a RAN CELL."""

    def __init__(self, num_units, input_size, keep_i=0.25, keep_h=0.75, depth=8, activation='tanh',
                 normalize=False, forget_bias=None):
        self.input_size = input_size
        self._num_units = num_units
        self.forget_bias = forget_bias

        self._activation = activation
        self._normalize = normalize

        self._keep_i = keep_i
        self._keep_h = keep_h
        self._depth = depth

    def zero_state(self, batch_size, dtype):
        noise_i = tf.random.uniform(shape=(batch_size, self.input_size), dtype=tf.float32) if self._keep_i < 1.0 else tf.ones(shape=(batch_size, self.input_size))
        noise_h = tf.random.uniform(shape=(batch_size, self._num_units), dtype=tf.float32) if self._keep_h < 1.0 else tf.ones(shape=(batch_size, self._num_units))
        return [tf.zeros((batch_size, self._num_units)), noise_i, noise_h]

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        current_state = state[0]
        noise_i = state[1]
        noise_h = state[2]

        for i in range(self._depth):
            if i == 0:
                h = _linear([inputs * noise_i], self._num_units, True, normalize=self._normalize)
            else:
                h = _linear([current_state * noise_h], self._num_units, True, normalize=self._normalize)

            t = tf.sigmoid(_linear([inputs * noise_i, current_state * noise_h], self._num_units, True,
                                    self.forget_bias, normalize=self._normalize)) if i == 0 else \
                tf.sigmoid(_linear([current_state * noise_h], self._num_units, True,
                                   self.forget_bias, normalize=self._normalize))

            current_state = (h - current_state) * t + current_state

        return current_state, [current_state, noise_i, noise_h]


class InterpretableRANStateTuple(object):
    __slots__ = ('c', '_i_list', '_f_list', 'w')

    def __init__(self, c, i_list, f_list, w_list):
        self.c = c
        self._i_list = i_list
        self._f_list = f_list
        self.w = w_list

    @classmethod
    def zero(cls, batch_size, state_size, dtype):
        return InterpretableRANStateTuple(
            c=tf.zeros([batch_size, state_size], dtype=dtype),
            i_list=[],
            f_list=[],
            w_list=[]
        )

    @classmethod
    def succeed(cls, c, i, f, prev: 'InterpretableRANStateTuple'):
        f_list = prev._f_list + [f]
        i_list = prev._i_list + [i]
        return InterpretableRANStateTuple(
            c=c,
            f_list=f_list,
            i_list=i_list,
            w_list=prev.w + InterpretableRANStateTuple._calc_weights(f=f_list, i=i_list)
        )

    @staticmethod
    def _calc_weights(i, f):
        t = len(i)
        w = np.zeros(t)
        for j in range(t):
            w[j] = i[j]
            for k in range(j + 1, t):
                w[j] *= f[k]
        return w

    @property
    def dtype(self):
        return self.c.dtype


class InterpretableSimpleRANCell(tf.keras.layers.Layer):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393).

    This is an implementation of the simplified RAN cell described in Equation group (2)."""

    def __init__(self, num_units, input_size=None, normalize=False):
        if input_size is not None:
            print(f"{self}: The input_size parameter is deprecated.")
        self._normalize = normalize
        self._num_units = num_units

    def zero_state(self, batch_size, dtype):
        return InterpretableRANStateTuple.zero(batch_size, self._num_units, dtype)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state: InterpretableRANStateTuple):
        with tf.variable_scope("ran_cell"):
            c = state.c
            value = tf.nn.sigmoid(_linear([c, inputs], 2 * self._num_units, True, normalize=self._normalize))
            i, f = tf.split(value=value, num_or_size_splits=2, axis=1)

            new_c = i * inputs + f * c

        return InterpretableRANStateTuple.succeed(new_c, i, f, state)
