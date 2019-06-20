import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lenet5_mnist_forward
import lenet5_mnist_backward
import numpy as np

TEST_INTERVAL_SECS = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples,
            lenet5_mnist_forward.IMAGE_SIZE,
            lenet5_mnist_forward.IMAGE_SIZE,
            lenet5_mnist_forward.NUM_CHANNELS
        ])
        y_ = tf.placeholder(tf.float32, [None, lenet5_mnist_forward.OUTPUT_NODE])
        y = lenet5_mnist_forward.forward((x, False, None))

        ema = tf.train.ExponentialMovingAverage(lenet5_mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(lenet5_mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_xs = np.reshape(mnist.test.images, (
                        mnist.test.num_examples,
                        lenet5_mnist_forward.IMAGE_SIZE,
                        lenet5_mnist_forward.IMAGE_SIZE,
                        lenet5_mnist_forward.NUM_CHANNELS
                    ))
                    accuracy_score = sess.run(accuracy, feed_dict={x:reshaped_xs, y_:mnist.test.labels})
                    print("After %s training step(s), test sccuracy = %g"%(global_step,accuracy_score))
                else:
                    print("No checkpoint file fount")
                    return
                time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./data/",one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()



