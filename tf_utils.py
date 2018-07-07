import tensorflow as tf

def get_static_shape(x):
    """Return the static shape of x"""
    x = tf.convert_to_tensor(x)
    return x.get_shape().as_list()

def get_dynamic_shape(x):
    x = tf.convert_to_tensor(x)
    return tf.unstack(tf.shape(x))

def get_shape(x):
    """Return static shape if available and dynamic shape if otherwise"""
    x = tf.convert_to_tensor(x)
    shapes = zip(get_static_shape(x), get_dynamic_shape(x))
    # Select the dynamic shape if static shape is unknown
    return [d if s is None else s for s, d in shapes]

def split_last_dimension(x, n):
    """Reshape x from [..., M] to [..., n, M // n]"""
    shape = get_static_shape(x)
    if shape[-1] % n != 0:
        raise ValueError('Last dimension ({}) must be divisible by n ({})'.format(
            shape[-1], n))
    return tf.reshape(x, shape[:-1] + [n, shape[-1] // n])

def combine_last_two_dimensions(x):
    """Reshape x from [..., N, M] to [..., M * N]"""
    shape = get_static_shape(x)
    return tf.reshape(x, shape[:-2] + [shape[-1] * shape[-2]])

