import math
import numpy as np
import collections


# inspired by https://github.com/amaas/stanford-ctc

class Hyp(object):
    """
    Container class for a single hypothesis
    """

    def __init__(self, pb, pnb, nc):
        """
        
        :param pb: probability of hypothesis ending in a blank character
        :param pnb: probability of hypothesis not ending in a blank character
        :param nc: number of characters in hypothesis
        """
        self.p_b = pb
        self.p_nb = pnb
        self.n_c = nc


class BeamLMDecoder:
    """
    Beam-search decoder with character LM
    """

    def __init__(self, lm) -> None:
        super().__init__()
        self.lm = lm

    @staticmethod
    def log_add(a: float, b: float, c=float('-inf')):
        psum = math.exp(a) + math.exp(b) + math.exp(c)
        if psum == 0.0:
            return float('-inf')
        else:
            return math.log(psum)

    def lm_score_final_char(self, prefix, next_char):
        """
        uses lm to score entire prefix
        returns only the log prob of final char
        
        :param prefix: the hypothis so far
        :param next_char: the next char in the hypothesis to get the probability for
        :return: probability of next_char added to hypothesis
        """
        # todo

        return None

    def decode(self, ctc_probs, beam_width=40, alpha=1.0, beta=0.0):

        """
        Decoder with an LM
        returns the best hypothesis in characters
        
        :param ctc_probs: [num_chars x seq_length] probabilities for characters from the ctc model
        :param beam_width: 
        :param alpha: language model weight
        :param beta: insertion bonus
        :return: (the best hypothesis in characters, probability of hypothesis (confidence measure?!) )
        """

        C = ctc_probs.shape[0]
        T = ctc_probs.shape[1]
        # change for loop if blank char != 0
        blank_char = 0

        key_fun = lambda x: self.log_add(x[1].p_nb, x[1].p_b) + beta * x[1].n_c
        default_init_fun = lambda: Hyp(float('-inf'), float('-inf'), 0)

        # current hypothesis
        H_curr = [[(), Hyp(float('-inf'), 0.0, 0)]]
        # old hypothesis
        # 'defaultdict' returns default value instead of None if item does not exist in dict yet
        H_old = collections.defaultdict(default_init_fun)

        for t in range(T):
            H_curr = dict(H_curr)
            H_next = collections.defaultdict(default_init_fun)

            for prefix, hyp in H_curr.items():

                H_next[prefix].p_b = self.log_add(hyp.p_nb + ctc_probs[blank_char, t],
                                                  hyp.p_b + ctc_probs[blank_char, t],
                                                  H_next[prefix].p_b)
                H_next[prefix].n_c = hyp.n_c
                if len(prefix) > 0:
                    H_next[prefix].p_nb = self.log_add(hyp.p_nb + ctc_probs[prefix[-1], t], H_next[prefix].p_nb)

                # for all chars that are not blank char
                for c in range(1, C):
                    new_prefix = tuple(list(prefix) + [c])

                    lm_prob = alpha * self.lm_score_final_char(prefix, c)

                    H_next[new_prefix].n_c = hyp.n_c + 1
                    if len(prefix) == 0 or (len(prefix) > 0 and c != prefix[-1]):
                        H_next[new_prefix].p_nb = self.log_add(hyp.p_nb + ctc_probs[c, t] + lm_prob,
                                                               hyp.p_b + ctc_probs[c, t] + lm_prob,
                                                               H_next[new_prefix].p_nb)
                    else:
                        H_next[new_prefix].p_nb = self.log_add(hyp.p_b + ctc_probs[c, t] + lm_prob,
                                                               H_next[new_prefix].p_nb)

                    if new_prefix not in H_curr:
                        H_next[new_prefix].p_b = self.log_add(H_old[new_prefix].p_nb + ctc_probs[blank_char, t],
                                                              H_old[new_prefix].p_b + ctc_probs[blank_char, t],
                                                              H_next[new_prefix].p_b)
                        H_next[new_prefix].p_nb = self.log_add(H_old[new_prefix].p_nb + ctc_probs[c, t],
                                                               H_next[new_prefix].p_nb)

            H_old = H_next
            H_curr = sorted(H_next.items(), key=key_fun, reverse=True)[:beam_width]

        # hyp = ''.join([self.int_char_map[i] for i in Hcurr[0][0]])
        hyp = H_curr[0][0]

        return hyp, key_fun(H_curr[0])
