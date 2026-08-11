"""Stage 8 — affordable adaptation — LoRA / quantization"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# facebook/nougat-base is a VisionEncoderDecoderModel: a DonutSwinModel encoder (attention
# Linear layers named "...attention.self.{query,key,value}", Swin/BERT convention) driving
# an MBartForCausalLM decoder (both self-attention AND cross-attention onto the image
# features, named "...self_attn.{q,k,v,out}_proj" / "...encoder_attn.{q,k,v,out}_proj",
# BART convention). Confirmed by loading the real pinned model and listing its named
# modules, not assumed from a generic Transformer diagram -- the two halves genuinely use
# different naming schemes, and hardcoding only one would silently apply LoRA to nothing
# on the other half.
DEFAULT_LORA_TARGET_KEYWORDS: tuple[str, ...] = (
    "query",
    "key",
    "value",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)


def _discover_lora_targets(model: Any, keywords: tuple[str, ...]) -> list[str]:
    """Suffix names of every Linear layer whose dotted name contains one of `keywords`.

    peft's `LoraConfig.target_modules` matches by module-name suffix, so this returns the
    short names (e.g. "q_proj"), not full paths -- passing full dotted paths would match
    nothing, since a different layer index appears in every path.
    """
    import torch.nn as nn

    found: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(kw in name for kw in keywords):
            found.add(name.rsplit(".", 1)[-1])
    return sorted(found)


def apply_lora(model: Any, cfg: dict) -> Any:
    """Wrap a component with LoRA per cfg.

    `cfg` is the `lora` block of `configs/train_ocr.yaml`: `r`, `alpha`, `dropout`, and an
    optional `target_module_keywords` override (default: DEFAULT_LORA_TARGET_KEYWORDS).
    Target module suffixes are discovered from the REAL model at call time (see
    `_discover_lora_targets`) rather than hardcoded, so this keeps working if a future
    `cfg["ocr"]["model"]` swap changes the underlying architecture's naming.

    Base model weights are frozen by `get_peft_model` itself (peft's documented behavior);
    this raises rather than silently training zero parameters if discovery finds nothing,
    since a fine-tune that trains nothing is a much worse failure than a loud one.
    """
    from peft import LoraConfig, get_peft_model

    keywords = tuple(cfg.get("target_module_keywords", DEFAULT_LORA_TARGET_KEYWORDS))
    target_modules = _discover_lora_targets(model, keywords)
    if not target_modules:
        raise ValueError(
            f"apply_lora: no Linear layers matched target_module_keywords={keywords} -- "
            "the model architecture may have changed; inspect model.named_modules()."
        )

    lora_cfg = LoraConfig(
        r=int(cfg["r"]),
        lora_alpha=int(cfg["alpha"]),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
    )
    wrapped = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    logger.info(
        f"adapt.apply_lora: {len(target_modules)} target module types "
        f"({', '.join(target_modules)}), {trainable:,}/{total:,} params trainable "
        f"({100 * trainable / total:.2f}%)"
    )
    return wrapped


def quantize(model: Any, cfg: dict) -> Any:
    """Post-training quantization per cfg. IMPLEMENT."""
    raise NotImplementedError("Adapt: quantize")
