from typing import List, NamedTuple, Dict
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # 2nd seminar code
        last_char = self.EMPTY_TOK
        res = []
        for ind in inds:
            char = self.ind2char[ind]
            if char == last_char:
                continue
            if char != self.EMPTY_TOK:
                res.append(char)
            last_char = ind
        return ''.join(res)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        # slightly modified 2nd seminar code
        dp = {
            ('', self.EMPTY_TOK): 1.0
        }
        for prob in probs:
            dp = self._extend_and_merge(dp, prob)
            dp = self._cut_beams(dp, beam_size)

        hypos: List[Hypothesis] = [Hypothesis(s.strip(), p) for (s, last_char), p in dp.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def _extend_and_merge(self, dp: dict, prob: torch.tensor) -> dict:
        new_dp = defaultdict(float)
        for (res, last_char), p in dp.items():
            for i in range(len(prob)):
                char = self.ind2char[i]
                if char == last_char:
                    new_dp[(res, last_char)] += p * prob[i]
                elif char == self.EMPTY_TOK:
                    new_dp[(res, char)] += p * prob[i]
                else:
                    new_dp[(res + char, char)] += p * prob[i]
        return new_dp

    @staticmethod
    def _cut_beams(self, dp: dict, beam_size: int) -> dict:
        return dict(list(sorted(dp.items(), key=lambda x: x[1], reverse=True))[:beam_size])