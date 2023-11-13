# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
import math
from math import sqrt, ceil, floor, log2, log
from itertools import chain, tee
from functools import lru_cache
import time
import random

import scipy.stats
from scipy.stats import chisquare, entropy, binom
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from normalizers import normalization_strategy_lookup
from alternative_prf_schemes import prf_lookup, seeding_scheme_lookup


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # simple default, find more schemes in alternative_prf_schemes.py
        select_green_tokens: bool = True,  # should always be the default if not running in legacy mode
        base: int = 2,  # base (radix) of each message
        message_length: int = 4,
        code_length: int = 4,
        use_position_prf: bool = True,
        use_fixed_position: bool = False,
        device: str = "cuda",
        **kwargs
    ):
        # patch now that None could now maybe be passed as seeding_scheme
        if seeding_scheme is None:
            seeding_scheme = "simple_1"
        self.device = device
        # Vocabulary setup
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None
        self._initialize_seeding_scheme(seeding_scheme)
        # Legacy behavior:
        self.select_green_tokens = select_green_tokens

        ### Parameters for multi-bit watermarking ###
        self.original_msg_length = message_length
        self.message_length = max(message_length, code_length)
        decimal = int("1" * message_length, 2)
        self.converted_msg_length = len(self._numberToBase(decimal, base))

        # if message bit width is leq to 2, no need to increase base
        if message_length <= 2:
            base = 2
        self.message = None
        self.bit_position = None
        self.base = base
        # self.chunk = int(ceil(log2(base)))
        assert floor(1 / self.gamma) >= base, f"Only {floor(1 / self.gamma)} chunks available " \
                                              f"with current gamma={self.gamma}," \
                                              f"But base is {self.base}"
        self.converted_message = None
        self.message_char = None
        self.use_position_prf = use_position_prf
        self.use_fixed_position = use_fixed_position
        self.bit_position_list = []
        self.position_increment = 0
        self.green_cnt_by_position = {i: [0 for _ in range(self.base)] for i
                                      in range(1, self.converted_msg_length + 1)}

    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(
            seeding_scheme
        )

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(
                f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG."
            )

        prf_key = prf_lookup[self.prf_type](
            input_ids[-self.context_width :], salt_key=self.hash_key
        )
        if self.use_position_prf:
            position_prf_key = prf_lookup["anchored_minhash_prf"](
                input_ids[-2:], salt_key=self.hash_key
            )
        else:
            position_prf_key = prf_key
        self.prf_key = prf_key

        # seeding for bit position
        random.seed(position_prf_key % (2**64 - 1))
        if self.use_fixed_position:
            self.bit_position = list(
                                range(1, self.converted_msg_length + 1)
                                )[self.position_increment % self.converted_msg_length]
        else:
            self.bit_position = random.randint(1, self.converted_msg_length)
        self.message_char = self.get_current_bit(self.bit_position)
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        candidate_greenlist = torch.chunk(vocab_permutation, floor(1 / self.gamma))
        return candidate_greenlist[self.message_char]

    def _get_colorlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        colorlist = torch.chunk(vocab_permutation, floor(1 / self.gamma))
        return colorlist

    # @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int, cand_msg=None):
        """Expensive re-seeding and sampling is cached."""
        ######################
        # self.converted_message = str(cand_msg) * self.converted_msg_length
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        # greenlist_ids = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        # return True if target in greenlist_ids else False, self.get_current_position()
        ######################
        colorlist_ids = self._get_colorlist_ids(torch.as_tensor(prefix, device=self.device))
        colorlist_flag = []
        for cl in colorlist_ids[:self.base]:
            if target in cl:
                colorlist_flag.append(True)
            else:
                colorlist_flag.append(False)

        return colorlist_flag, self.get_current_position()

    def get_current_bit(self, bit_position):
        # embedding stage
        if self.converted_message:
            return int(self.converted_message[bit_position - 1])
        # extraction stage
        else:
            return 0

    def get_current_position(self):
        return self.bit_position

    def set_message(self, binary_msg: str = ""):
        self.message = binary_msg
        self.converted_message = self._convert_binary_to_base(binary_msg)

    def _convert_binary_to_base(self, binary_msg: str):
        decimal = int(binary_msg, 2)
        converted_msg = self._numberToBase(decimal, self.base)
        converted_msg = "0" * (self.converted_msg_length - len(converted_msg)) + converted_msg
        return converted_msg

    def _numberToBase(self, n, b):
        """
        https://stackoverflow.com/a/28666223
        """
        if n == 0:
            return str(0)
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return "".join(map(str, digits[::-1]))
    def flush_position(self):
        positions = "".join(list(map(str, self.bit_position_list)))
        self.bit_position_list = []
        self.green_cnt_by_position = {i: [0 for _ in range(self.base)] for i
                                      in range(1, self.converted_msg_length + 1)}
        return [positions]


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores in between model outputs and next token sampler.
    """

    def __init__(self, *args, store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None
        self.feedback = kwargs.get('use_feedback', False)
        self.feedback_args = kwargs.get('feedback_args', {})
        if self.store_spike_ents:
            self._init_spike_entropies()

    def _init_spike_entropies(self):
        alpha = torch.exp(torch.tensor(self.delta)).item()
        gamma = self.gamma

        self.z_value = ((1 - gamma) * (alpha - 1)) / (1 - gamma + (alpha * gamma))
        self.expected_gl_coef = (gamma * alpha) / (1 - gamma + (alpha * gamma))

        # catch for overflow when bias is "infinite"
        if alpha == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0

    def _get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def _get_and_clear_stored_spike_ents(self):
        spike_ents = self._get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def _compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1 + (self.z_value * probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def _calc_greenlist_mask(
        self, scores: torch.FloatTensor, greenlist_token_ids
    ) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(
        self, scores: torch.Tensor, colorlist_mask: torch.Tensor, greenlist_bias: float,
            denylist_flag=False
    ) -> torch.Tensor:
        if denylist_flag:
            scores[colorlist_mask] = 0
        else:
            scores[colorlist_mask] = scores[colorlist_mask] + greenlist_bias
        return scores

    def _score_rejection_sampling(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_compute"
    ) -> list[int]:
        """Generate greenlist based on current candidate next token. Reject and move on if necessary. Method not batched.
        This is only a partial version of Alg.3 "Robust Private Watermarking", as it always assumes greedy sampling. It will still (kinda)
        work for all types of sampling, but less effectively.
        To work efficiently, this function can switch between a number of rules for handling the distribution tail.
        These are not exposed by default.
        """
        sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)

        final_greenlist = []
        for idx, prediction_candidate in enumerate(greedy_predictions):
            greenlist_ids = self._get_greenlist_ids(
                torch.cat([input_ids, prediction_candidate[None]], dim=0)
            )  # add candidate to prefix
            if prediction_candidate in greenlist_ids:  # test for consistency
                final_greenlist.append(prediction_candidate)

            # What follows below are optional early-stopping rules for efficiency
            if tail_rule == "fixed_score":
                if sorted_scores[0] - sorted_scores[idx + 1] > self.delta:
                    break
            elif tail_rule == "fixed_list_length":
                if len(final_greenlist) == 10:
                    break
            elif tail_rule == "fixed_compute":
                if idx == 40:
                    break
            else:
                pass  # do not break early
        return torch.as_tensor(final_greenlist, device=input_ids.device)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""

        # this is lazy to allow us to co-locate on the watermarked model's device
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng
        feedback_bias = self.feedback_args.get("feedback_bias", -1)

        #TODO: batchify ecc with feedback
        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        list_of_blacklist_ids = [[] for _ in input_ids]
        feedback_flag = False
        for b_idx, input_seq in enumerate(input_ids):
            if self.self_salt:
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            list_of_greenlist_ids[b_idx] = greenlist_ids
            if self.use_fixed_position:
                self.position_increment += 1
            self.bit_position_list.append(self.bit_position)

            # logic for computing and storing spike entropies for analysis
            if self.store_spike_ents:
                if self.spike_entropies is None:
                    self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self._compute_spike_entropy(scores[b_idx]))

            # logic for whether to expand the greenlist and suppress blacklist
            if self.feedback:
                # increment the colorlist for the current token
                context_length = self.context_width + 1 - self.self_salt
                if input_seq.shape[-1] < context_length:
                    continue
                ngram = input_seq[-context_length:]
                ngram = tuple(ngram.tolist())
                target = ngram[-1]
                prefix = ngram if self.self_salt else ngram[:-1]
                colorlist_ids = self._get_colorlist_ids(torch.as_tensor(prefix, device=self.device))
                pos = self.get_current_position()
                for c_idx, cl in enumerate(colorlist_ids[:self.base]):
                    if target in cl:
                        self.green_cnt_by_position[pos][c_idx] += 1
                colorlist_flag, pos = self._get_ngram_score_cached(prefix, target)

                eta = self.feedback_args.get("eta", 3)
                tau = self.feedback_args.get("tau", 2)
                msg = int(self.converted_message[pos-1])
                preliminary_cond = sum(self.green_cnt_by_position[pos]) >= eta
                if preliminary_cond:
                    max_color = np.argmax(self.green_cnt_by_position[pos])
                    cond_1 = max_color != msg
                    colorlist_ids = list(colorlist_ids)
                    if cond_1:
                        # feedback_flag = True
                        list_of_blacklist_ids[b_idx] = colorlist_ids[max_color]
                        continue
                    if tau == -1:
                        continue
                    top2_color = np.argpartition(self.green_cnt_by_position[pos], -2)[-2]
                    color_cnt_diff = self.green_cnt_by_position[pos][max_color] - \
                                     self.green_cnt_by_position[pos][top2_color]

                    cond_2 = color_cnt_diff < tau + 1
                    if cond_2:
                        list_of_blacklist_ids[b_idx] = colorlist_ids[top2_color]


        green_tokens_mask = self._calc_greenlist_mask(
            scores=scores, greenlist_token_ids=list_of_greenlist_ids
        )
        scores = self._bias_greenlist_logits(
            scores=scores, colorlist_mask=green_tokens_mask,
            greenlist_bias=self.delta
        )
        if self.feedback:
            # suppress the black list when condition is satisfied
            black_tokens_mask = self._calc_greenlist_mask(
                scores=scores, greenlist_token_ids=list_of_blacklist_ids
            )
            scores = self._bias_greenlist_logits(
                scores=scores, colorlist_mask=black_tokens_mask,
                greenlist_bias=feedback_bias, denylist_flag=True
            )

        ## hardlisting for debugging ###
        # scores[~green_tokens_mask] = 0
        ##

        return scores


class WatermarkDetector(WatermarkBase):
    """This is the detector for all watermarks imprinted with WatermarkLogitsProcessor.

    The detector needs to be given the exact same settings that were given during text generation  to replicate the watermark
    greenlist generation and so detect the watermark.
    This includes the correct device that was used during text generation, the correct tokenizer, the correct
    seeding_scheme name, and parameters (delta, gamma).

    Optional arguments are
    * normalizers ["unicode", "homoglyphs", "truecase"] -> These can mitigate modifications to generated text that could trip the watermark
    * ignore_repeated_ngrams -> This option changes the detection rules to count every unique ngram only once.
    * z_threshold -> Changing this threshold will change the sensitivity of the detector.
    """

    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_ngrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def dummy_detect(
        self,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_all_window_scores: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        return_bit_match: bool = True,
        return_z_score_max: bool = True
    ):
        # HF-style output dictionary
        score_dict = dict()
        score_dict.update(dict(pred_message=""))
        score_dict.update({'min_pos_ratio': float("nan")})
        score_dict.update({'max_pos_ratio': float("nan")})
        score_dict.update(dict(custom_metric=float("nan")))
        score_dict.update(dict(sampled_positions=""))
        score_dict.update(dict(position_acc=float("nan")))
        score_dict.update(dict(bit_match=float("nan")))
        score_dict.update(dict(cand_match=float("nan")))
        score_dict.update(dict(cand_acc=float("nan")))
        score_dict.update(dict(cand_acc_2=float("nan")))
        score_dict.update(dict(cand_acc_4=float("nan")))
        score_dict.update(dict(cand_acc_8=float("nan")))
        score_dict.update(dict(cand_acc_ablation=float("nan")))
        score_dict.update(dict(cand_match_ablation=float("nan")))
        score_dict.update(dict(decoding_time=float("nan")))
        score_dict.update(dict(confidence_per_position=[]))
        score_dict.update(dict(error_pos=[]))

        # if return_z_score_max:
        #     score_dict.update(dict(z_score_max=float("nan")))

        if return_bit_match:
            score_dict.update(dict(bit_acc=float("nan")))
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=float("nan")))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=float("nan")))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=float("nan")))
        if return_z_score:
            score_dict.update(dict(z_score=float("nan")))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = float("nan")
            score_dict.update(dict(p_value=float("nan")))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=[]))
        if return_all_window_scores:
            score_dict.update(dict(window_list=[]))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=torch.tensor([])))

        output_dict = {}
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = False

        return output_dict


    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        return_bit_match: bool = True,
        return_z_score_max: bool =True,
        message: str = "",
        **kwargs,
    ):
        s_time = time.time()
        gold_message = message
        ########## sequential extraction #########
        if True:
            ngram_to_watermark_lookup, frequencies_table, ngram_to_position_lookup, green_cnt_by_position, \
                position_list = self._score_ngrams_in_passage_sequential(input_ids)
        else:
            ngram_to_watermark_lookup, frequencies_table, ngram_to_position_lookup, green_cnt_by_position \
                = self._score_ngrams_in_passage(input_ids)
            # dummy positions
            position_list = [0,0,0,0]
        ##############################################
        # count positions for all tokens
        position_cnt = {}
        for k, v in ngram_to_position_lookup.items():
            freq = frequencies_table[k]
            position_cnt[v] = position_cnt.get(v, 0) + freq

        # compute confidence per position
        p_val_per_position = []
        for p in range(1, self.converted_msg_length + 1):
            all_green_cnt = np.array(green_cnt_by_position[p])
            green_cnt = max(all_green_cnt)
            if position_cnt.get(p) is None:
                position_cnt[p] = 0
            T = position_cnt.get(p)
            # binom_pval = self._compute_binom_p_val(green_cnt, T)
            multi_pval = self._compute_max_multinomial_p_val(green_cnt, T)
            p_val_per_position.append(multi_pval)

        # predict message
        list_decoded_msg, ran_list_decoded_msg, elapsed_time = \
            self._predict_message(position_cnt, green_cnt_by_position, p_val_per_position)
        elapsed_time = elapsed_time - s_time
        # compute bit accuracy
        best_prediction = list_decoded_msg[0]
        correct_bits, total_bits, error_pos = self._compute_ber(best_prediction, gold_message)
        prediction_results = {'confidence': [], 'random': []}
        # for our list decoded msg
        for msg in list_decoded_msg:
            cb, tb, _ = self._compute_ber(msg, gold_message)
            prediction_results['confidence'].append(cb)
        # for random list decoded msg
        for msg in ran_list_decoded_msg:
            cb, tb, _ = self._compute_ber(msg, gold_message)
            prediction_results['random'].append(cb)

        # for debugging
        if False:
            if kwargs['col_name'] == "w_wm_output":
                print(gold_message)
                # if matched_bits != total_bits:
                #     print(kwargs['text'])
                print(correct_bits / total_bits)
                # for err in error_pos:
                #     print(green_cnt_by_position[err // 2 + 1])
                #     print(gold_message[err])
                # print(min(position_cnt.values()))
                # print(sum(position_cnt.values()))
                print(position_cnt)
                print(sum(position_cnt.values()))
                print(green_cnt_by_position)
                breakpoint()

        # use the predicted message to select ngram_to_watermark_lookup
        for ngram, green_token in ngram_to_watermark_lookup.items():
            pos = ngram_to_position_lookup[ngram]
            msg = best_prediction[pos - 1]
            ngram_to_watermark_lookup[ngram] = ngram_to_watermark_lookup[ngram][msg]

        green_token_mask, green_unique, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )

        # Count up scores over all ngrams
        if self.ignore_repeated_ngrams:
            # Method that only counts a green/red hit once per unique ngram.
            # New num total tokens scored (T) becomes the number unique ngrams.
            # We iterate over all unique token ngrams in the input, computing the greenlist
            # induced by the context in each, and then checking whether the last
            # token falls in that greenlist.
            # compute it again in case
            position_cnt = {}
            for k, v in ngram_to_position_lookup.items():
                position_cnt[v] = position_cnt.get(v, 0) + 1
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt, "Error for num_tokens_scored"
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(
                    frequencies_table.values(), ngram_to_watermark_lookup.values()
                )
            )

        # assert green_token_count == green_token_count_debug, "Debug: green_token_count != green_token_count_debug"

        assert green_token_count == green_unique.sum()
        # HF-style output dictionary
        score_dict = dict()
        sampled_positions = "" if len(position_list) == 0 else "".join(list(map(str, position_list)))
        score_dict.update(dict(sampled_positions=sampled_positions))
        score_dict.update(dict(pred_message="".join(map(str, best_prediction))))
        min_val = min(position_cnt.values())
        max_val = max(position_cnt.values())
        sum_val = sum(position_cnt.values())
        score_dict.update({'min_pos_ratio': min_val / sum_val})
        score_dict.update({'max_pos_ratio': max_val / sum_val})
        score_dict.update(dict(custom_metric=-sum(p_val_per_position)))
        score_dict.update(dict(decoding_time=elapsed_time))
        score_dict.update(dict(confidence_per_position=p_val_per_position))
        score_dict.update(dict(error_pos=error_pos))
        if return_bit_match:
            score_dict.update(dict(bit_acc=correct_bits / total_bits))
            score_dict.update(dict(bit_match=correct_bits == total_bits))
            score_dict.update(dict(cand_match=max(prediction_results['confidence']) == total_bits))
            score_dict.update(dict(cand_match_ablation=max(prediction_results['random']) == total_bits))
            score_dict.update(dict(cand_acc=max(prediction_results['confidence']) / total_bits))
            score_dict.update(dict(cand_acc_2=max(prediction_results['confidence'][:3]) / total_bits))
            score_dict.update(dict(cand_acc_4=max(prediction_results['confidence'][:5]) / total_bits))
            score_dict.update(dict(cand_acc_8=max(prediction_results['confidence'][:9]) / total_bits))
            score_dict.update(dict(cand_acc_ablation=max(prediction_results['random']) / total_bits))
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(
                dict(z_score=z_score)
            )
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask.tolist()))
        if return_z_at_T:
            # Score z_at_T separately:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))
            score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict

    def _compute_z_score(self, observed_count, T):
        """
        count refers to number of green tokens, T is total number of tokens
        If T <= 0, this means this position was not sampled. Return 0 for this.
        """
        if T <= 0:
            return 0
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_binom_p_val(self, observed_count, T):
        """
        count refers to number of green tokens, T is total number of tokens
        If T <= 0, this means this position was not sampled. Return p-val=1 for this.
        """
        if T <= 0:
            return 1
        binom_p = self.gamma
        observed_count -= 1
        # p value for observing a sample geq than the observed count
        p_val = 1 - binom.cdf(max(observed_count, 0), T, binom_p)
        return p_val

    def _compute_max_multinomial_p_val(self, observed_count, T):
        """
        Compute the p-value by subtracting the cdf(observed_count -1) of multinomial~(T, 1/base, ... 1/base),
        which is the probability of observing a sample as extreme or more as the observed_count
        The computation follows from Levin, Bruce. "A representation for multinomial cumulative distribution functions."
        The Annals of Statistics (1981): 1123-1126.
        """
        if T <= 0:
            return 1
        poiss = scipy.stats.poisson
        normal = scipy.stats.norm
        k = self.base
        s = T
        a = observed_count - 1
        poiss_cdf_X = poiss.cdf(a, T / k)
        normal_approx_W = normal.cdf(0.5 / np.sqrt(T)) - normal.cdf(-0.5 / np.sqrt(T))
        log_max_multi_cdf = math.log(np.sqrt(2 * math.pi * T)) + k * math.log(poiss_cdf_X) + math.log(normal_approx_W)
        max_multi_cdf = math.exp(log_max_multi_cdf)
        p_val = 1 - min(1, max_multi_cdf)
        return p_val

    def _compute_hoeffdings_bound(self, observed_count, T):
        """
        Compute bound using Hoeffding's inequality.
        Similar to using the normal approximation to the binomial
        """
        if T <= 0:
            return 1
        mean = T / self.base
        delta = max(0, observed_count - mean)
        bound = math.exp(-2 * delta ** 2 / T)
        return bound


    def _predict_message(self, position_cnt, green_cnt_by_position, p_val_per_pos,
                         num_candidates=16):
        s_time = 0
        msg_prediction = []
        confidence_per_pos = []
        for pos in range(1, self.converted_msg_length + 1):
            # find the index (digit) with the max counts of colorlist
            p_val = p_val_per_pos[pos-1]
            if position_cnt.get(pos) is None: # no allocated tokens (may happen when T / b is small)
                position_cnt[pos] = -1
                preds = random.sample(list(range(self.base)), 2)
                pred = preds[0]
                next_idx = preds[1]
                confidence_per_pos.append((-p_val, pred, next_idx, pos))

            else:
                green_counts = green_cnt_by_position[pos]
                pred, val = max(enumerate(green_counts), key=lambda x: (x[1], x[0]))
                sorted_idx = np.argsort(green_counts)
                max_idx, next_idx = sorted_idx[-1], sorted_idx[-2]
                confidence_per_pos.append((-p_val, max_idx, next_idx, pos))
            msg_prediction.append(pred)

        elapsed_time = time.time() - s_time
        random_prediction_list = [msg_prediction]

        # sample random bits
        cnt = 0
        while cnt < num_candidates:
            msg_decimal = random.getrandbits(self.message_length)
            converted_msg = self._numberToBase(msg_decimal, self.base)
            converted_msg = "0" * (self.converted_msg_length - len(converted_msg)) + converted_msg
            random_prediction_list.append(converted_msg)
            cnt += 1

        msg_prediction_list = [msg_prediction]
        num_candidate_position = ceil(log2(num_candidates + 1))
        # sort by the least confident positions
        confidence_per_pos = sorted(confidence_per_pos, key=lambda x: x[0])[:num_candidate_position]
        cnt = 0
        candidate_iter = iter(powerset(confidence_per_pos))
        while cnt < num_candidates:
            try:
                candidate = next(candidate_iter)
            except:
                break
            cand_msg = msg_prediction.copy()
            for _, max_idx, next_idx, pos in candidate:
                cand_msg[pos - 1] = next_idx
            msg_prediction_list.append(cand_msg)
            cnt += 1
        # print(msg_prediction)
        # print(len(msg_prediction_list))
        # breakpoint()
        return msg_prediction_list, random_prediction_list, elapsed_time

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    # @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int, cand_msg=None):
        """Expensive re-seeding and sampling is cached."""
        ######################
        # self.converted_message = str(cand_msg) * self.converted_msg_length
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        # greenlist_ids = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        # return True if target in greenlist_ids else False, self.get_current_position()
        ######################
        colorlist_ids = self._get_colorlist_ids(torch.as_tensor(prefix, device=self.device))
        colorlist_flag = []
        for cl in colorlist_ids[:self.base]:
            if target in cl:
                colorlist_flag.append(True)
            else:
                colorlist_flag.append(False)

        return colorlist_flag, self.get_current_position()

    def _score_ngrams_in_passage(self, input_ids: torch.Tensor):
        """Core function to gather all ngrams in the input and compute their watermark."""
        if len(input_ids) - self.context_width < 1:
            raise ValueError(
                f"Must have at least {1} token to score after "
                f"the first min_prefix_len={self.context_width} tokens required by the seeding scheme."
            )

        # Compute scores for all ngrams contexts in the passage:
        token_ngram_generator = ngrams(
            input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt
        )
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = collections.defaultdict(list)
        ngram_to_position_lookup = {}
        # initialize dictionary of {position: [0, ..., r]}
        green_cnt_by_position = {i: [0 for _ in range(self.base)] for i in range(1, self.converted_msg_length + 1)}

        for idx, ngram_example in enumerate(frequencies_table.keys()):
            prefix = ngram_example if self.self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            colorlist_flag, current_position = self._get_ngram_score_cached(prefix, target)
            ngram_to_watermark_lookup[ngram_example] = colorlist_flag
            ngram_to_position_lookup[ngram_example] = current_position
            for cand_msg, flag in enumerate(colorlist_flag):
                if flag:
                    green_cnt_by_position[current_position][cand_msg] += frequencies_table[ngram_example]
            ##############################
            # for cand_msg in range(self.base):
            #     greenlist_flag, current_position = self._get_ngram_score_cached(prefix, target, cand_msg)
            #     ngram_to_watermark_lookup[ngram_example].append(greenlist_flag)
            #     position is chosen independently of the message content
            #     so looping through the message will not change the current position
                # ngram_to_position_lookup[ngram_example] = current_position
                # mark the color list
                # if greenlist_flag:
                #     green_cnt_by_position[current_position][cand_msg] += frequencies_table[ngram_example]
            ##############################

        return ngram_to_watermark_lookup, frequencies_table, ngram_to_position_lookup, green_cnt_by_position

    def _score_ngrams_in_passage_sequential(self, input_ids: torch.Tensor):
        position_list = []
        frequencies_table = {}
        ngram_to_position_lookup = {}
        ngram_to_watermark_lookup = {}
        green_cnt_by_position = {i: [0 for _ in range(self.base)] for i in range(1, self.converted_msg_length + 1)}
        increment = self.context_width - self.self_salt
        # loop through tokens to get the sampled positions
        for idx in range(self.context_width, len(input_ids) + self.self_salt):
            pos = increment % self.converted_msg_length + 1
            ngram = input_ids[idx - self.context_width: idx + 1 - self.self_salt]
            ngram = tuple(ngram.tolist())
            frequencies_table[ngram] = frequencies_table.get(ngram, 0) + 1
            target = ngram[-1]
            prefix = ngram if self.self_salt else ngram[:-1]
            if self.use_fixed_position:
                colorlist_flag, _ = self._get_ngram_score_cached(prefix, target)
            else:
                colorlist_flag, pos = self._get_ngram_score_cached(prefix, target)
            for f_idx, flag in enumerate(colorlist_flag):
                if flag:
                    # if kwargs['col_name'] == "w_wm_output":
                        # print(f"PRF: {self.prf_key}")
                        # print(f"position: {pos}")
                        # print(f"colorlist: {f_idx}")
                        # breakpoint()
                        # print("\n\n")
                    green_cnt_by_position[pos][f_idx] += 1
            ngram_to_watermark_lookup[ngram] = colorlist_flag
            position_list.append(pos)
            ngram_to_position_lookup[ngram] = pos
            increment += 1

        return ngram_to_watermark_lookup, frequencies_table, ngram_to_position_lookup, green_cnt_by_position, \
            position_list



    def _get_green_at_T_booleans(self, input_ids, ngram_to_watermark_lookup) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate binary list of green vs. red per token, a separate list that ignores repeated ngrams, and a list of offsets to
        convert between both representations:
        green_token_mask = green_token_mask_unique[offsets] except for all locations where otherwise a repeat would be counted
        """
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt)

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (torch.tensor(green_token_mask),
                torch.tensor(green_token_mask_unique),
                torch.tensor(offsets),
                )

    def _score_windows_impl_batched(
        self,
        input_ids: torch.Tensor,
        window_size: str,
        window_stride: int = 1,
    ):
        # Implementation details:
        # 1) --ignore_repeated_ngrams is applied globally, and windowing is then applied over the reduced binary vector
        #      this is only one way of doing it, another would be to ignore bigrams within each window (maybe harder to parallelize that)
        # 2) These windows on the binary vector of green/red hits, independent of context_width, in contrast to Kezhi's first implementation
        # 3) z-scores from this implementation cannot be directly converted to p-values, and should only be used as labels for a
        #    ROC chart that calibrates to a chosen FPR. Due, to windowing, the multiple hypotheses will increase scores across the board#
        #    naive_count_correction=True is a partial remedy to this

        ngram_to_watermark_lookup, frequencies_table = self._score_ngrams_in_passage(input_ids)
        green_mask, green_ids, offsets = self._get_green_at_T_booleans(
            input_ids, ngram_to_watermark_lookup
        )
        len_full_context = len(green_ids)

        partial_sum_id_table = torch.cumsum(green_ids, dim=0)

        if window_size == "max":
            # could start later, small window sizes cannot generate enough power
            # more principled: solve (T * Spike_Entropy - g * T) / sqrt(T * g * (1 - g)) = z_thresh for T
            sizes = range(1, len_full_context)
        else:
            sizes = [int(x) for x in window_size.split(",") if len(x) > 0]

        z_score_max_per_window = torch.zeros(len(sizes))
        cumulative_eff_z_score = torch.zeros(len_full_context)
        s = window_stride

        window_fits = False
        for idx, size in enumerate(sizes):
            if size <= len_full_context:
                # Compute hits within window for all positions in parallel:
                window_score = torch.zeros(len_full_context - size + 1, dtype=torch.long)
                # Include 0-th window
                window_score[0] = partial_sum_id_table[size - 1]
                # All other windows from the 1st:
                window_score[1:] = partial_sum_id_table[size::s] - partial_sum_id_table[:-size:s]

                # Now compute batched z_scores
                batched_z_score_enum = window_score - self.gamma * size
                z_score_denom = sqrt(size * self.gamma * (1 - self.gamma))
                batched_z_score = batched_z_score_enum / z_score_denom

                # And find the maximal hit
                maximal_z_score = batched_z_score.max()
                z_score_max_per_window[idx] = maximal_z_score

                z_score_at_effective_T = torch.cummax(batched_z_score, dim=0)[0]
                cumulative_eff_z_score[size::s] = torch.maximum(
                    cumulative_eff_z_score[size::s], z_score_at_effective_T[:-1]
                )
                window_fits = True  # successful computation for any window in sizes

        if not window_fits:
            raise ValueError(
                f"Could not find a fitting window with window sizes {window_size} for (effective) context length {len_full_context}."
            )

        # Compute optimal window size and z-score
        cumulative_z_score = cumulative_eff_z_score[offsets]
        optimal_z, optimal_window_size_idx = z_score_max_per_window.max(dim=0)
        optimal_window_size = sizes[optimal_window_size_idx]
        return (
            optimal_z,
            optimal_window_size,
            z_score_max_per_window,
            cumulative_z_score,
            green_mask,
        )

    def _score_sequence_window(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        window_size: str = None,
        window_stride: int = 1,
    ):
        (
            optimal_z,
            optimal_window_size,
            _,
            z_score_at_T,
            green_mask,
        ) = self._score_windows_impl_batched(input_ids, window_size, window_stride)

        # HF-style output dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=optimal_window_size))

        denom = sqrt(optimal_window_size * self.gamma * (1 - self.gamma))
        green_token_count = int(optimal_z * denom + self.gamma * optimal_window_size)
        green_fraction = green_token_count / optimal_window_size
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=green_fraction))
        if return_z_score:
            score_dict.update(dict(z_score=optimal_z))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=z_score_at_T))
        if return_p_value:
            z_score = score_dict.get("z_score", optimal_z)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))

        # Return per-token results for mask. This is still the same, just scored by windows
        # todo would be to mark the actually counted tokens differently
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_mask.tolist()))

        return score_dict

    def _compute_ber(self, pred_msg: list, message: str):
        pred_msg = "".join(map(str, pred_msg))
        decimal = int(pred_msg, self.base)
        decimal = min(decimal, 2 ** self.message_length - 1)
        binary_pred = format(decimal, f"0{self.message_length}b")
        use_ecc = False
        if use_ecc:
            rm = reedmuller.ReedMuller(2, 5)
            decoded = rm.decode(list(map(int, binary_pred)))
            print(decoded)
            if decoded:
                binary_pred = ''.join(map(str, decoded))
            else:
                binary_pred = binary_pred[:self.original_msg_length]

        # predicted binary message is longer because the last chunk was right-padded
        if len(binary_pred) != len(message):
            print(f"Predicted msg: {pred_msg}")
            print(f"Predicted binary msg: {binary_pred}")
            print(f"Gold msg: {message}")
            breakpoint()
            raise RuntimeError("Extracted message length is shorter the original message!")

        match = 0
        total = 0
        error_pos = []
        for pos, (g, p) in enumerate(zip(message, binary_pred)):
            if g == p:
                match += 1
                error_pos.append(False)
            else:
                error_pos.append(True)
            total += 1
        return match, total, error_pos

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> dict:
        """Scores a given string of text and returns a dictionary of results."""
        assert (text is not None) ^ (
            tokenized_text is not None
        ), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs[
                "return_p_value"
            ] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        # call score method
        output_dict = {}

        if window_size is not None:
            # assert window_size <= len(tokenized_text) cannot assert for all new types
            score_dict = self._score_sequence_window(
                tokenized_text,
                window_size=window_size,
                window_stride=window_stride,
                **kwargs,
            )
            output_dict.update(score_dict)
        else:
            kwargs['text'] = text
            score_dict = self._score_sequence(tokenized_text, **kwargs)

        if return_scores:
            output_dict.update(score_dict)
            # score sampled positions
            gold_position = kwargs['position'][self.context_width - self.self_salt:]
            position = score_dict['sampled_positions']
            match_cnt = sum([x == y for x, y in zip(gold_position, position)])
            output_dict.update(dict(position_acc=match_cnt / len(position)))

        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        # convert any numerical values to float if requested
        if convert_to_float:
            for key, value in output_dict.items():
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict


##########################################################################
# Ngram iteration from nltk, extracted to remove the dependency
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
##########################################################################


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))