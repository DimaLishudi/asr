from typing import List, NamedTuple, Dict
from collections import defaultdict

from multiprocessing import Pool
from itertools import repeat
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

    def ctc_beam_search_batch(self, batch_probs: torch.tensor, batch_probs_length,
                              beam_size: int = 100, num_workers=2) -> List[List[Hypothesis]]:
        with Pool(num_workers) as p:
            res = p.starmap(self.ctc_beam_search, zip(batch_probs, batch_probs_length, repeat(beam_size)))
        return res

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
        for i in range(probs_length-1):
            dp = self._extend_and_merge(dp, probs[i])
            dp = self._cut_beams(dp, beam_size)

        # last iteration: merge only depending on resulting text, not last character
        dp = self._extend_and_merge_final(dp, probs[probs_length-1])

        hypos: List[Hypothesis] = [Hypothesis(s, p) for s, p in dp.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)[:beam_size]

    def _extend_and_merge(self, dp: dict, prob: torch.tensor) -> dict:
        new_dp = defaultdict(float)
        for (res, last_char), p in dp.items():
            for i in range(len(prob)):
                char = self.ind2char[i]
                if char == last_char:
                    new_dp[(res, char)] += p * prob[i]
                elif char == self.EMPTY_TOK:
                    new_dp[(res, char)] += p * prob[i]
                else:
                    new_dp[(res + char, char)] += p * prob[i]
        return new_dp

    def _extend_and_merge_final(self, dp: dict, prob: torch.tensor) -> dict:
        new_dp = defaultdict(float)
        for (res, last_char), p in dp.items():
            for i in range(len(prob)):
                char = self.ind2char[i]
                if char == last_char:
                    new_dp[res.strip()] += p * prob[i]
                elif char == self.EMPTY_TOK:
                    new_dp[res.strip()] += p * prob[i]
                else:
                    new_dp[(res + char).strip()] += p * prob[i]
        return new_dp

    @staticmethod
    def _cut_beams(dp: dict, beam_size: int) -> dict:
        return dict(list(sorted(dp.items(), key=lambda x: x[1], reverse=True))[:beam_size])