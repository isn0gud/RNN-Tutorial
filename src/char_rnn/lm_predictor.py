import os

from six.moves import cPickle
from char_rnn.model import Model
import tensorflow as tf


class LMPredictor:
    def __init__(self, save_dir) -> None:
        super().__init__()

        self.save_dir = save_dir

        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            self.saved_args = cPickle.load(f)
        with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
            self.chars, self.vocab = cPickle.load(f)

        self.model = Model(self.saved_args, training=False)

    def prob_next_char(self, prefix, char):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                prob_next_char = self.model.prob_next_char(sess, self.chars, self.vocab, char,
                                                           prefix)
                return prob_next_char

                # print('Probability for [{}->{}] == {}'
                #       .format(args.prime, args.next, prob_next_char))
