from typing import *

import tinygrad
from tinygrad import Tensor
from tinygrad import dtypes
import numpy as np
from tinygrad import TinyJit

from utils import COLORS


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


class SamplerConfig:
    """
    Encapsulation of all available sampler hyperparameters.

    This should be a good starting point for baselining experiments.
    """
    temp: float = 0.666
    top_p: float = 0.95
    top_k: int = 27
    min_p: float = 0.03  # Turn this down to 0.01 to reduce the shoggoth

    low_ent_thresh: float = 7.0
    low_vent_thresh: float = 7.0
    med_ent_thresh: float = 10.0
    med_vent_thresh: float = 10.0
    high_ent_thresh: float = 13.0
    high_vent_thresh: float = 13.0
    
    # Attention Entropy Thresholds
    low_attention_entropy_threshold: float = 11.915
    medium_attention_entropy_threshold: float = 11.921
    high_attention_entropy_threshold: float = 11.926

    # Attention Varentropy Thresholds
    low_attention_varentropy_threshold: float = 0.001
    medium_attention_varentropy_threshold: float = 0.0045
    high_attention_varentropy_threshold: float = 0.009

    # Agreement Thresholds
    low_agreement_threshold: float = 2e-06
    medium_agreement_threshold: float = 4e-06
    high_agreement_threshold: float = 5e-06

    # Interaction Strength Thresholds
    low_interaction_strength_threshold: float = 0.2
    medium_interaction_strength_threshold: float = 0.247
    high_interaction_strength_threshold: float = 0.264

    # TODO this is a bit of a nasty mess, but also makes all the hyperparameters visible
    helv_attn_ent_offset: float = 1.3
    helv_attn_ent_coef: float = 0.2

    lehv_interaction_strength_offset: float = 1.2
    lehv_interaction_strength_coef: float = 0.3

    hehv_attn_ent_coef: float = 0.2
    hehv_attn_vent_offset: float = 2.0
    hehv_attn_vent_coef: float = 0.5

    # TODO not convinced this should
    n_adaptive_samples: int = 5

    # Adaptive sampling parameters
    ada_temp_logits: float = 0.3
    ada_temp_attn: float = 0.2
    ada_temp_agree: float = 0.2
    ada_top_p: float = 0.1
    ada_top_k_int: float = 0.3
    ada_top_k_agree: float = 0.2
    ada_min_p: float = 0.5
    ada_score_logits_ent: float = 0.1
    ada_score_attn_ent: float = 0.2
    ada_score_logits_vent: float = 0.3
    ada_score_attn_vent: float = 0.4
    ada_score_agree: float = 0.5
    ada_score_int: float = 0.6



# pure function, compile
@TinyJit
def calculate_varentropy_logsoftmax(logits: Tensor, axis: int = -1) -> Tuple[Tensor, Tensor]:
    """
    Entropy and varentropy from logits using log softmax function.
    """    
    log_probs = Tensor.log_softmax(logits, axis=axis)
    probs = Tensor.exp(log_probs)
    entropy = -Tensor.sum(probs * log_probs, axis=axis) / LN_2
    varentropy = Tensor.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, axis=axis)
    return entropy, varentropy


def multinominal_sample_one(probs_sort: Tensor, key: int) -> Tensor:
    """
    Samples one token from a multinomial distribution with sorted probabilities.
    """
    # mlx does not have the exponential random distribution
    # but we can model it using the uniform distribution like below
    # taking 1 - u to move the domain of log to (0, 1] instead of [0, 1)
    Tensor.manual_seed(key)
    np.random.seed(key)
    u = Tensor.rand(shape=probs_sort.shape)
    q = -Tensor.log(Tensor.negative(u) + 1)
    return Tensor.argmax(probs_sort / q, axis=-1, keepdims=True).cast(dtypes.int32)


def flip(x: Tensor, axis: int = -1):
    """
    Reverse the order of elements along a given axis
    """
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(None, None, -1)
    return x[tuple(slices)]


def calculate_metrics(logits: Tensor, attention_scores: Tensor) -> Dict[str, Tensor]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = Tensor.softmax(attention_scores, axis=-1)
    attn_entropy = -Tensor.sum(attention_probs * Tensor.log2(Tensor.clip(attention_probs, 1e-7, 1.0)), axis=-1)
    attn_varentropy = Tensor.var(attn_entropy, axis=1)

    mean_attention = Tensor.mean(attention_probs, axis=1)
    agreement = Tensor.mean(Tensor.abs(attention_probs - mean_attention.unsqueeze(1)), axis=(1, 2))

    interaction_strength = Tensor.mean(Tensor.abs(attention_scores), axis=(1, 2, 3))

    return dict(
        logits_entropy=Tensor.mean(entropy),
        logits_varentropy=Tensor.mean(varentropy),
        attn_entropy=Tensor.mean(attn_entropy),
        attn_varentropy=Tensor.mean(attn_varentropy),
        agreement=Tensor.mean(agreement),
        interaction_strength=interaction_strength
    )


def _in1d(element: Tensor, test_elements: Tensor, invert: bool = False) -> Tensor:
    arr1, arr2 = element.flatten(), test_elements.flatten()
    if arr1.size == 0 or arr2.size == 0:
        return Tensor.ones(arr1.shape, dtype=dtypes.bool) if invert else Tensor.zeros(arr1.shape, dtype=dtypes.bool)
    if invert:
        return (arr1.unsqueeze(-1) != arr2.unsqueeze(0)).all(-1)
    return (arr1.unsqueeze(-1) == arr2.unsqueeze(0)).any(-1)


def isin(element: Tensor, test_elements: Tensor, invert: bool = False) -> Tensor:
    """
    hacky isin function to mimic `jax.numpy.isin`
    """
    ele = Tensor(element) if not isinstance(element, Tensor) else element
    tele = Tensor(test_elements) if not isinstance(test_elements, Tensor) else test_elements
    result = _in1d(ele, tele, invert=invert)
    return result.reshape(element.shape)


def _sample(
        logits: Tensor, *, temperature: Union[float, Tensor], top_p: Union[float, Tensor], 
        top_k: Union[int, Tensor], min_p: Union[float, Tensor], key: int = None
    ) -> Tensor:
    if key is None:
        key = 1337

    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = Tensor.softmax(logit / temperature, axis=-1)
    
    # apply min_p sampling
    if min_p > 0.0:
        p_max = Tensor.max(probs, axis=-1, keepdims=True)
        indices_to_remove = probs < (min_p * p_max)
        replacement = Tensor.ones_like(logit) * float('-inf')
        logit = Tensor.where(indices_to_remove, replacement, logit)

    # apply top-k sampling
    _indices = np.argsort(-probs, axis=-1)
    top_k_indices = _indices[:, :top_k]
    top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)
    probs_sort = flip(top_k_probs, axis=-1)
    probs_idx = flip(top_k_indices, axis=-1)
    probs_sum = Tensor(np.cumsum(probs_sort, axis=-1))

    # apply top_p sampling
    mask = Tensor.where(probs_sum - probs_sort > top_p, 1.0, 0.0)
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / Tensor.sum(probs_sort, axis=-1, keepdims=True)
    next_token = multinominal_sample_one(probs_sort, key)
    next_token_g = Tensor.take_along_axis(probs_idx, next_token.reshape(bsz, 1), axis=-1)
    return next_token_g.cast(dtypes.int32)


# our hero
def sample(
        gen_tokens: Tensor, logits: Tensor, attention_scores: Tensor, cfg: SamplerConfig,
        clarifying_question_token: int = 2564, key: int = None
    ) -> Tuple[Tensor, str, dict]:
    if key is None:
        key = 1337
    
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # low entropy, low varentropy = "flowing with unspoken intent"
    # the model is very certain, choose the most likely token
    if (ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh and 
        attn_ent < cfg.low_attention_entropy_threshold and
        attn_vent < cfg.low_attention_varentropy_threshold and
        agreement < cfg.low_agreement_threshold and
        interaction_strength < cfg.low_interaction_strength_threshold):
        return Tensor.argmax(logits[:, -1], axis=-1, keepdims=True).cast(dtypes.int32), COLORS["lelv"], metrics
    
    # high entropy, low varentropy = "treading carefully, asking clarifying questions"
    # the model is uncertain but consistently so, leading to careful sampling or 
    # asking clarifying questions.
    elif (ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent < cfg.low_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength < cfg.low_interaction_strength_threshold):
        if not isin(gen_tokens[:, -1], clarifying_question_token).any():
            return Tensor([[clarifying_question_token]]), COLORS["hehv"], metrics
        else:
            # if we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * attn_ent
            return _sample(logits, temperature=min(1.5, cfg.temp * temp_adj), top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p, key=key), COLORS["helv"], metrics
        
    # low entropy, high varentropy: "exploring forks in the path"
    elif (ent < cfg.low_ent_thresh and vent > cfg.high_vent_thresh and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength > cfg.low_interaction_strength_threshold):
        print("(lehv)", flush = True, end = "")
        temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        return _sample(logits, temperature=min(1.5, cfg.temp * temp_adj), top_p=cfg.top_p, top_k=top_k_adj, min_p=cfg.min_p, key=key), COLORS["lehv"], metrics

    # high entropy, high varentropy: "resampling in the mist"
    elif (ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh and 
          attn_ent > cfg.high_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          agreement > cfg.high_agreement_threshold and
          interaction_strength > cfg.high_interaction_strength_threshold):
        print("(hehv)", flush = True, end = "")
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * attn_ent)  # Decrease top_p when attention entropy is high
        return _sample(logits, temperature=max(2.0, cfg.temp * temp_adj), top_p=top_p_adj, top_k=cfg.top_k, min_p=cfg.min_p, key=key), COLORS["hehv"], metrics
    
    # middle ground: use adaptive sampling
    else:
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = cfg.temp * (1 + cfg.ada_temp_logits * ent + cfg.ada_temp_attn * attn_ent - cfg.ada_temp_agree * metrics["agreement"])
        top_p = Tensor.clip(cfg.top_p * (1 + cfg.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
        top_k = Tensor.clip(
            Tensor.round(cfg.top_k * (1 + cfg.ada_top_k_int * metrics["interaction_strength"].item() - cfg.ada_top_k_agree * metrics["agreement"].item())),
            a_min=1,
            a_max=100
        ).cast(dtypes.uint32).item()
        min_p = Tensor.clip(cfg.min_p * (1 - cfg.ada_min_p * vent), 0.01, 0.5)

        keys = [key + i for i in range(cfg.n_adaptive_samples)]

        # basically, sample n(5) number of times
        # choose the best from it
        samples = []
        for sample_key in keys:
            sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, key=sample_key)
            samples.append(sample)

        def score_sample(sample: Tensor):
            bsz, seqlen = sample.shape
            vbsz = logits.shape[-1]
            one_hot = Tensor.zeros((bsz, seqlen, vbsz))
            one_hot[Tensor.arange(bsz)[:, None], Tensor.arange(seqlen)[None, :], sample] = 1
            log_prob = Tensor.sum(Tensor.log_softmax(logits) * one_hot)
            confidence_score = (
                (1 - ent / cfg.high_ent_thresh) * cfg.ada_score_logits_ent +
                (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.ada_score_attn_ent +
                (1 - vent / cfg.high_vent_thresh) * cfg.ada_score_logits_vent +
                (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.ada_score_attn_vent +
                (agreement / cfg.high_agreement_threshold) * cfg.ada_score_agree +
                (interaction_strength / cfg.high_interaction_strength_threshold) * cfg.ada_score_int
            )
            return log_prob + confidence_score
        
        sample_scores = [score_sample(sample) for sample in samples]
        best_sample_idx = Tensor.argmax(Tensor(sample_scores)).item()
        return samples[best_sample_idx], COLORS["ada"], metrics
