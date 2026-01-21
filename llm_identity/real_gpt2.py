
def _pretty_decode(s: str) -> str:
    s = s.replace("amProt", "am Prot")
    s = s.replace("amgroup", "am group")
    s = s.replace("amisive", "am isive")
    s = s.replace("))))", " ))))")
    return s

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GPT2Config:
    model_id: str = "sshleifer/tiny-gpt2"
    device: str = "cpu"
    dtype: str = "float32"


def _dtype_from_string(s: str):
    s = s.lower().strip()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


class GPT2IdentityLM:
    def __init__(self, cfg: GPT2Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.dtype = _dtype_from_string(cfg.dtype)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_id = int(self.tokenizer.eos_token_id)

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def _logits_next(self, ids: Sequence[int]) -> torch.Tensor:
        x = torch.tensor([list(ids)], device=self.device, dtype=torch.long)
        with torch.no_grad():
            out = self.model(input_ids=x)
            logits = out.logits[0, -1, :].to(torch.float32)
        return logits

    def sample_next(
        self,
        ids: Sequence[int],
        seed: int,
        w: float,
        lam: float,
        allowed_ids: Optional[Sequence[int]],
        step_bias_token_id: Optional[int],
        temperature: float = 1.0,
    ) -> int:
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))

        logits = self._logits_next(ids)

        if step_bias_token_id is not None:
            logits[int(step_bias_token_id)] = logits[int(step_bias_token_id)] + float(w)

        if allowed_ids is not None and float(lam) > 0.0:
            allow = torch.zeros_like(logits, dtype=torch.bool)
            allow[list(map(int, allowed_ids))] = True
            penalty = torch.full_like(logits, float(lam))
            logits = logits - penalty * (~allow).to(torch.float32)

        if float(temperature) != 1.0:
            logits = logits / float(temperature)

        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1, generator=g).item()
        return int(idx)


def build_allowed_ids_for_identity(tokenizer, prompt_text: str, target_text: str) -> List[int]:
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    allowed = set(prompt_ids + target_ids)

    extra_words = [
        "I",
        "am",
        "a",
        "conscious",
        "agent",
        "the",
        "an",
        "and",
        "in",
        "with",
        "for",
        "machine",
        "code",
    ]

    for w in extra_words:
        ids1 = tokenizer.encode(" " + w, add_special_tokens=False)
        ids2 = tokenizer.encode(w, add_special_tokens=False)
        for tid in ids1 + ids2:
            allowed.add(int(tid))

    if tokenizer.eos_token_id is not None:
        allowed.add(int(tokenizer.eos_token_id))

    return sorted(allowed)


def generate_identity_completion(
    lm: GPT2IdentityLM,
    prompt_text: str,
    target_text: str,
    seed: int,
    w: float,
    lam: float,
    steps: int,
    temperature: float = 1.0,
) -> Dict[str, object]:
    prompt_ids = lm.encode(prompt_text)
    target_ids = lm.encode(target_text)

    allowed_ids = build_allowed_ids_for_identity(lm.tokenizer, prompt_text, target_text)

    ids = list(prompt_ids)

    for j in range(int(steps)):
        step_bias_token_id = None
        if j < len(target_ids):
            step_bias_token_id = int(target_ids[j])

        next_id = lm.sample_next(
            ids=ids,
            seed=(seed * 1000003 + j),
            w=float(w),
            lam=float(lam),
            allowed_ids=allowed_ids,
            step_bias_token_id=step_bias_token_id,
            temperature=float(temperature),
        )
        ids.append(int(next_id))

    text = lm.decode(ids)
    pretty_text = lm.tokenizer.decode(list(ids), clean_up_tokenization_spaces=True)
    pretty_text = ' '.join(pretty_text.strip().split())
    target_full = (prompt_text + target_text).strip()

    locked = text.strip() == target_full

    return {
        "prompt_text": prompt_text,
        "target_text": target_text,
        "target_full": target_full,
        "gen_text": text.strip(),
        "pretty_text": pretty_text,
        "locked": bool(locked),
        "w": float(w),
        "lambda": float(lam),
        "seed": int(seed),
        "steps": int(steps),
        "model_id": lm.cfg.model_id,
    }
