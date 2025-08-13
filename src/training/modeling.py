"""
모델/토크나이저 로딩 및 QLoRA/양자화 보조 유틸리티 모듈
"""

from __future__ import annotations

import subprocess
from typing import Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def maybe_install_flash_attn() -> tuple[str, torch.dtype]:
    """가능하면 flash-attn 설치 후 권장 dtype 반환.

    Returns:
        (attn_impl, torch_dtype)
    """

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        subprocess.run(["pip", "install", "-qqq", "flash-attn"], check=False)
        return "flash_attention_2", torch.bfloat16
    return "eager", torch.float16


def build_quant_config(load_in_4bit: bool, torch_dtype: torch.dtype):
    """QLoRA용 양자화 설정을 생성합니다."""

    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=False,
    )


def load_model_and_tokenizer(
    model_id_key: str,
    model_dict: dict[str, str],
    *,
    num_labels: int,
    quant_config,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """모델 및 토크나이저 로드.

    Args:
        model_id_key: 모델 딕셔너리 키
        model_dict: 키→허브 경로 매핑
        num_labels: 분류 클래스 수
        quant_config: bnb 양자화 설정 (없을 수 있음)
    """

    hub_path = model_dict[model_id_key]

    model = AutoModelForSequenceClassification.from_pretrained(
        hub_path,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map={"": 0} if torch.cuda.is_available() else None,
        num_labels=num_labels,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(hub_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

