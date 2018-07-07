import tensorflow as tf
import tf_utils

def split_heads(x, num_heads):
    """Split and transpose fom [B, L, E] into [B, num_heads, L, E // num_heads]"""
    return tf.transpose(tf_utils.split_last_dimension(x, num_heads), [0, 2, 1, 3])

def combine_heads(x):
    """Inverse of split_heads"""
    return tf_utils.combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def scaled_dot_product_attention(Q, K, V, dropout_rate=0.0):
    scaler = tf.rsqrt(tf.to_float(tf_utils.get_shape(Q)[2])) # depth of the query
    logits = tf.matmul(Q, K, transpose_b=True) * scaler
    weights = tf.nn.softmax(logits)
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    return tf.matmul(weights, V)

def multihead_attention(query, memory, key_depth, value_depth, output_depth, num_heads):
    # Linearly project each of the query and memory
    Q = tf.layers.dense(query, key_depth, use_bias=False)
    K = tf.layers.dense(memory, key_depth, use_bias=False)
    V = tf.layers.dense(memory, value_depth, use_bias=False)

    # Split into seperate heads
    Q = split_heads(Q, num_heads)
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    # Attention
    x = scaled_dot_product_attention(Q, K, V)

    # Recombine and project
    x = combine_heads(x)
    x = tf.layers.dense(x, output_depth, use_bias=False)
    return x

def feed_forward(x, hidden_units, output_depth):
    x = tf.layers.conv1d(x, filters=hidden_units, kernel_size=1, activation=tf.nn.relu, use_bias=True)
    x = tf.layers.conv1d(x, filters=output_depth, kernel_size=1, activation=None, use_bias=True)
    return x
