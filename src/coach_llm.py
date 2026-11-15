"""
Optional "writer" for the coach line (see coach.py): a small local language model (Qwen2.5-0.5B-Instruct, Apache-2.0, models/coach_llm/) that
rephrases the sentence the rules already chose, so the tips do not all sound the same.

The model never decides WHAT to say: the rules (coach.py) pick the advice from the measurements, the model only gives it new wording.
Anything doubtful is thrown away and the rule's own sentence is used instead:
  - the model, its libraries or the GPU/CPU run are missing or fail            -> rule sentence
  - the answer is not one short plain-English line, contains digits the rule sentence did not have, talks about "camera"/"data"/AI,
    is much longer/shorter than it or keeps too few of its content words, or turns praise into criticism (or the other way round)      -> rule sentence
FOOTBALL_COACH_LLM=0 switches the writer off. Runs offline: nothing is fetched at run time.
"""
import os
import re
import time
import zlib

from app_paths import BUNDLE_ROOT
from logging_utils import log

MODEL_DIR = os.path.join(BUNDLE_ROOT, "models", "coach_llm", "Qwen2.5-0.5B-Instruct")
ENABLED = os.environ.get("FOOTBALL_COACH_LLM", "1") != "0"

SYSTEM = ("You rewrite football coaching tips for an amateur player. Say the SAME tip with different words: keep every idea of the original, "
          "in the same order, about the same length, speaking to the player as 'you'. Change only the wording. "
          "Never add ideas, numbers, places or players, never give the opposite advice. Answer with the rewritten tip only.")
EXAMPLES = [
    ("Clean ball to a free teammate - that is the pattern to repeat; try to play it a touch earlier next time.",
     "That pass reached a free teammate, which is exactly the pattern to repeat; next time try to release it a touch sooner."),
    ("A soft pass gives defenders time to step in - play it firmly along the ground.",
     "If a pass is too soft, defenders get time to step in, so play it firmly along the ground."),
    ("You had 1.4 s of free space - lift your head earlier, a forward pass or a shot may have been on.",
     "There was 1.4 s of open space in front of you, so look up sooner: a forward pass or a shot could have been available."),
    ("Taking the shot on is right; pick the spot early and strike it with the laces.",
     "Going for goal was the right call; choose your spot early and hit it with your laces."),
]
BAD_WORDS = re.compile(r"camera|footage|video|data|table|measure|percent|\bai\b|language model|sorry|as an|cannot|can't|\bhere'?s\b", re.I)
PRAISE = re.compile(r"\b(good|clean|nice|great|tidy|crisp|well|right idea|fair call|composure)\b", re.I)
BLAME = re.compile(r"\b(bad|poor|wrong|mistake|lost|lose|failed|should have|didn't|did not|too )\b", re.I)


def _stems(text):
    return {w[:5] for w in re.findall(r"[a-z]{5,}", text.lower())}


def _valid(src, out):
    if not out or "\n" in out or not (35 <= len(out) <= 260) or not re.fullmatch(r"[\x20-\x7E]+", out):
        return False
    if BAD_WORDS.search(out) or re.search(r"\d", out) and not re.search(r"\d", src):
        return False
    a, b = _stems(src), _stems(out)
    if not (0.6 <= len(out) / len(src) <= 1.5) or len(a & b) < max(2, int(0.35 * len(a))):
        return False
    if bool(PRAISE.search(src)) and not BLAME.search(src) and BLAME.search(out):
        return False
    if bool(BLAME.search(src)) and not PRAISE.search(src) and PRAISE.search(out) and not BLAME.search(out):
        return False
    return True


class CoachWriter:
    """Loads the model on first use (a few seconds), keeps it for the rest of the process; `rewrite` never raises."""

    def __init__(self):
        self.model = self.tok = self.device = None
        self.failed = not ENABLED or not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))
        self.n_ok = self.n_fallback = 0

    def _load(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()
        log(f"CoachWriter: language model loaded on {self.device} in {time.time() - t0:.1f} s.")

    def rewrite(self, sentence, move, seed):
        """The rule sentence in new words, or None (the caller then keeps the rule sentence)."""
        if self.failed:
            return None
        try:
            if self.model is None:
                self._load()
            import torch
            msgs = [{"role": "system", "content": SYSTEM}]
            for src, dst in EXAMPLES:
                msgs += [{"role": "user", "content": f"Tip: {src}"}, {"role": "assistant", "content": dst}]
            msgs.append({"role": "user", "content": f"Tip: {sentence}"})
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = {k: v.to(self.device) for k, v in ids.items()}
            torch.manual_seed(zlib.crc32(seed.encode()))      # same move -> same wording on every re-render
            with torch.no_grad():
                out = self.model.generate(**ids, max_new_tokens=60, do_sample=True, temperature=0.35, top_p=0.9,
                                          pad_token_id=self.tok.eos_token_id)
            text = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"').strip()
            if _valid(sentence, text):
                self.n_ok += 1
                return text
        except Exception as exc:   # noqa: BLE001 - the writer is optional, the rule sentence is always there
            log(f"CoachWriter: switched off for this run ({exc!r}).")
            self.failed = True
            return None
        self.n_fallback += 1
        return None

    def close(self):
        if self.model is not None:
            log(f"CoachWriter: {self.n_ok} sentences rewritten, {self.n_fallback} kept in the rule wording.")
            self.model = self.tok = None
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:   # noqa: BLE001
                pass
