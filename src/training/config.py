"""
파인튜닝 설정 모듈

- 본 모듈은 파인튜닝에 필요한 설정 값을 데이터클래스로 정의합니다.
- 각 설정은 명시적으로 타입을 지정하여 가독성과 안전성을 높입니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class ModelConfig:
    """모델 관련 설정 값.

    Attributes:
        model_id: 사용할 모델 식별자 키 (예: "Llama-3.1-8B-Instruct").
        model_dict: 모델 키와 HF 허브 경로 매핑.
        num_labels: 분류 클래스 수.
        use_qlora: QLoRA 사용 여부.
        load_in_4bit: 4비트 로딩 사용 여부.
        use_flash_attn_if_available: 가능 시 flash-attn 사용 여부.
    """

    model_id: str
    model_dict: Dict[str, str]
    num_labels: int = 4
    use_qlora: bool = True
    load_in_4bit: bool = True
    use_flash_attn_if_available: bool = True


@dataclass
class DataConfig:
    """데이터 관련 설정 값."""

    train_csv_path: Path
    eval_csv_path: Path
    test_csv_path: Path
    text_field: str = "EMR"
    raw_label_field: str = "True"
    max_seq_length: int = 1024


@dataclass
class MLflowConfig:
    """MLflow 관련 설정 값."""
    
    experiment_name: str = "cde-eeg-llm-finetuning"
    tracking_uri: str = "sqlite:///mlruns.db"  # 로컬 SQLite DB
    artifact_location: str = "mlruns"
    log_artifacts: bool = True
    log_models: bool = True
    log_hyperparams: bool = True
    log_metrics: bool = True


@dataclass
class TrainerConfig:
    """트레이너/학습 관련 설정 값."""

    output_dir: Path
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    fp16: bool = True
    bf16: bool = False
    optim: str = "paged_adamw_8bit"
    save_steps: int = 25
    logging_steps: int = 10
    max_grad_norm: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.1
    group_by_length: bool = True
    lr_scheduler_type: str = "cosine"
    report_to: str = "mlflow"
    gradient_checkpointing: bool = True
