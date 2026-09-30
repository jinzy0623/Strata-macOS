"""Apple Metal/ggml implementation via llama-cpp-python (MIT).

This is a complete small-GGUF inference path, not a translation of the CUDA
expert cache. ggml supplies ARM NEON/Accelerate, Metal kernels and mmap.
"""
from __future__ import annotations

import platform
import time
from pathlib import Path

from serve.backends import BackendBundle
from serve.frontend import ChatTemplate


class GGUFTokenizer:
    def __init__(self, model):
        self.model = model

    def encode(self, text, parse_special=False):
        return self.model.tokenize(text.encode("utf-8"), add_bos=False, special=parse_special)

    def decode(self, ids, errors="replace"):
        return self.model.detokenize(ids, special=True).decode("utf-8", errors=errors)


class GGMLTemplate:
    def __init__(self, model, source):
        self.template = ChatTemplate(source=source)
        self.bos = model.detokenize([model.token_bos()], special=True).decode("utf-8", "replace")
        self.eos = model.detokenize([model.token_eos()], special=True).decode("utf-8", "replace")

    @staticmethod
    def starts_in_reasoning(prompt):
        return prompt.rfind("<think>") > prompt.rfind("</think>")

    def render(self, messages, tools=None, **kwargs):
        # Strata's tool XML parser is Qwen3.8-specific. Do not promise arbitrary
        # GGUF tool schemas or vision on the first Apple backend.
        if tools:
            raise ValueError("GGUF backend currently supports text chat only; tools are not supported")
        return self.template.render(messages, tools=None, bos_token=self.bos, eos_token=self.eos, **kwargs)


class GGMLBackend:
    def __init__(self, model, *, metal):
        self.model = model
        self.max_context = model.n_ctx()
        self.info = {"backend": "metal" if metal else "ggml-cpu", "version": "0.1.13-macos.1",
                     "architecture": platform.machine(), "expert_streaming": False}
        self.last = {}

    def generate(self, ids, max_new, sampling, cancel):
        options = {"temp": sampling.get("temperature", 0.8), "top_p": sampling.get("top_p", 0.95),
                   "top_k": sampling.get("top_k", 40), "min_p": sampling.get("min_p", 0.05),
                   "repeat_penalty": sampling.get("repetition_penalty", 1.0),
                   "frequency_penalty": sampling.get("frequency_penalty", 0.0),
                   "presence_penalty": sampling.get("presence_penalty", 0.0)}
        if sampling.get("seed") is not None:
            self.model.set_seed(int(sampling["seed"]))
        if cancel.is_set():
            return
        started, first, count = time.monotonic(), None, 0
        gen = self.model.generate(ids, reset=True, **options)
        try:
            for token in gen:
                if cancel.is_set():
                    break
                if first is None:
                    first = time.monotonic()
                count += 1
                yield token
                if count >= max_new:
                    break
        finally:
            gen.close()
            ended = time.monotonic()
            self.last = {"generated": count, "prompt_tokens": len(ids),
                         "prompt_ms": ((first or ended) - started) * 1000,
                         "decode_ms": (ended - (first or ended)) * 1000}

    def close(self):
        self.model.close()


def create_backend(config, *, metal=True):
    if metal and (platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64")):
        raise ValueError("Metal requires native ARM64 Python on Apple Silicon macOS; avoid Rosetta")
    path = Path(config.get("model", ""))
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError("config.model must be an existing GGUF file")
    context = int(config.get("context", 4096))
    if context < 128:
        raise ValueError("context must be at least 128")
    # Avoid loading a known impossible configuration. Leave room for macOS and KV.
    import psutil
    available = psutil.virtual_memory().available
    if path.stat().st_size > available * 0.8:
        raise ValueError("GGUF is too large for available unified memory; select a smaller quantized model")
    from llama_cpp import Llama, llama_cpp
    if metal and not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("llama-cpp-python has no GPU backend; rebuild with setup-macos.sh")
    model = Llama(model_path=str(path.resolve()), n_ctx=context,
                  n_batch=min(512, context), n_gpu_layers=-1 if metal else 0,
                  use_mmap=True, use_mlock=False, verbose=True)
    try:
        source = model.metadata.get("tokenizer.chat_template")
        if not source:
            raise ValueError("GGUF has no tokenizer.chat_template; use an instruction/chat GGUF with an embedded template")
        tok = GGUFTokenizer(model)
        stops = {model.token_eos()}
        # Only singleton special ids are terminators. Encoding a missing special
        # into ordinary pieces must never turn normal letters into stop tokens.
        for special in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>"):
            ids = tok.encode(special, parse_special=True)
            if len(ids) == 1:
                stops.add(ids[0])
        return BackendBundle(GGMLBackend(model, metal=metal), tok, GGMLTemplate(model, source), stops)
    except BaseException:
        model.close()
        raise
