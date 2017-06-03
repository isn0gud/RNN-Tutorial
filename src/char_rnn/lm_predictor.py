import os

from six.moves import cPickle
from char_rnn.model import Model
import tensorflow as tf


class LModel:
    def __init__(self, save_dir) -> None:
        super().__init__()

        self.save_dir = save_dir

        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            self.saved_args = cPickle.load(f)
        with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
            self.chars, self.vocab = cPickle.load(f)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = Model(self.saved_args, training=False)
            self.sess = tf.Session()
            glob_init = tf.global_variables_initializer()
            self.sess.run(glob_init)
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def prob_next_char(self, prefix, char):
        #     Build a graph containing `net1`.
        if prefix == "":
            # todo this should be handled by the language model to return the most probable chars that begin a sequence
            # if ther is no prefix return equal prob for each char
            return 1.0 / 29.0
        else:
            with self.graph.as_default():
                prob_next_char = self.model.prob_next_char(self.sess, self.chars, self.vocab, char,
                                                           prefix)
                return prob_next_char

                # print('Probability for [{}->{}] == {}'
                #       .format(args.prime, args.next, prob_next_char))

    def prob_next_chars(self, prefix):
        #     Build a graph containing `net1`.
        if prefix == "":
            # todo this should be handled by the language model to return the most probable chars that begin a sequence
            # if ther is no prefix return equal prob for each char
            ret = dict(map(lambda idx_n_char_prob: (self.chars[idx_n_char_prob[0]], idx_n_char_prob[1]),
                           enumerate([1.0 / 29.0] * 28)))
            # beginning with space prob = 0
            ret[0] = 0
            return ret
        else:
            with self.graph.as_default():
                prob_next_chars = self.model.prob_next_chars(self.sess, self.chars, self.vocab, prefix)
                return prob_next_chars

                # print('Probability for [{}->{}] == {}'
                #       .format(args.prime, args.next, prob_next_char))


if __name__ == '__main__':
    predictor = LModel("data/dev-clean-50k-night/checkpoints")
    print(predictor.prob_next_char("HELLO MY NAME IS", " "))
    print(predictor.prob_next_char("HELLO MY NAME IS", " "))
    print(predictor.prob_next_char("HELLO MY NAME IS ", " "))
