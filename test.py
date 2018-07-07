import tensorflow as tf
import attention

class TestAttention(tf.test.TestCase):
    def testAttention(self):
        # Batches, Sentence length, Embedding size
        N, L, E = 3, 5, 32

        q = tf.random_normal([N, L, E])
        m = tf.random_normal([N, L, E])
        x = attention.multihead_attention(q, m, 32, 32, 32, 8)
        x = attention.feed_forward(x, 256, 32)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result = x.eval()

if __name__ == '__main__':
    tf.test.main()
