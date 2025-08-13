"""
데이터 전처리 및 데이터셋 유틸리티 모듈

- CSV를 로드하여 허깅페이스 datasets 포맷으로 변환하고, 토크나이저를 이용해 입력 텐서를 생성합니다.
- 레이블 문자열을 다중분류 정수 레이블로 변환합니다.
"""

from __future__ import annotations

from typing import Dict

from datasets import load_dataset


LABEL_TO_ID: Dict[str, int] = {
    "Normal": 0,
    "Abnormal(Nonspecific)": 1,
    "Abnormal(Interictal)": 2,
    "Abnormal(Ictal)": 3,
}


def convert_labels_to_multiclass(examples: Dict) -> Dict:
    """문자 레이블을 정수 레이블로 변환합니다."""

    label_text = examples.get("labels")
    examples["labels"] = LABEL_TO_ID.get(label_text, -1)
    return examples


def preprocess_function(examples: Dict, tokenizer, text_field: str, max_length: int) -> Dict:
    """텍스트 필드 토크나이징 및 레이블 부착."""

    inputs = tokenizer(
        examples[text_field],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs["labels"] = examples["labels"]
    return inputs


def load_and_prepare_datasets(
    train_csv_path: str,
    eval_csv_path: str,
    tokenizer,
    *,
    text_field: str,
    raw_label_field: str,
    max_length: int,
):
    """
    CSV 경로로부터 train/eval 데이터셋을 로드하고 토크나이징을 수행합니다.
    """

    train_dataset = load_dataset("csv", data_files=str(train_csv_path), split="train")
    eval_dataset = load_dataset("csv", data_files=str(eval_csv_path), split="train")

    # 레이블 컬럼 이름 통일
    train_dataset = train_dataset.rename_column(raw_label_field, "labels")
    eval_dataset = eval_dataset.rename_column(raw_label_field, "labels")

    # 문자열 → 정수 레이블 변환
    train_dataset = train_dataset.map(convert_labels_to_multiclass)
    eval_dataset = eval_dataset.map(convert_labels_to_multiclass)

    # 토크나이징
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, text_field, max_length), batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, text_field, max_length), batched=True
    )

    return train_dataset, eval_dataset

