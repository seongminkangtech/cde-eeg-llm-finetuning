"""
Hugging Face Transformers + TRL 기반 파인튜닝 트레이너
"""

from __future__ import annotations

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from datasets import load_dataset
import mlflow
import mlflow.pytorch

from .config import ModelConfig, DataConfig, TrainerConfig
from .data import load_and_prepare_datasets
from .modeling import (
    maybe_install_flash_attn,
    build_quant_config,
    load_model_and_tokenizer,
)


def cleanup_memory():
    """메모리 정리 함수 (GPU/CPU 모두 적용)"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()


def predict_label(pred):
    """
    예측된 레이블 인덱스를 실제 레이블로 변환합니다.
    
    Args:
        pred (torch.Tensor): 예측된 레이블 인덱스.
    
    Returns:
        str: 실제 레이블.
    """
    # 상수 정의
    LABELS = ['Normal', 'Abnormal(Nonspecific)', 'Abnormal(Interictal)', 'Abnormal(Ictal)']
    return LABELS[int(pred[0])]


def test_model(model, tokenizer, text):
    """
    주어진 텍스트에 대해 모델을 사용하여 예측을 수행합니다.
    
    Args:
        model: 예측에 사용할 모델.
        tokenizer: 텍스트를 토크나이징하는 데 사용할 토크나이저.
        text (str): 예측할 텍스트.
    
    Returns:
        torch.Tensor: 예측된 레이블 인덱스.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions


def evaluate_model_on_test_data(eval_df, model, tokenizer, text_field: str):
    """
    주어진 데이터프레임의 각 행에 대해 모델을 평가하고 예측 레이블을 추가합니다.
    각 반복마다 메모리 정리를 수행합니다.
    
    Args:
        eval_df (pd.DataFrame): 평가할 데이터프레임.
        model: 평가에 사용할 모델.
        tokenizer: 텍스트를 토크나이징하는 데 사용할 토크나이저.
        text_field (str): 텍스트 컬럼 이름.
    
    Returns:
        pd.DataFrame: 예측 레이블이 추가된 데이터프레임.
    """
    print("테스트 데이터셋에서 모델 성능을 평가합니다...")
    
    for idx, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Evaluating"):
        try:
            pred = test_model(model, tokenizer, row[text_field])
            eval_df.at[idx, 'Pred'] = predict_label(pred)
        except Exception as e:
            print(f"[{idx}] 예외 발생: {e}")
            eval_df.at[idx, 'Pred'] = None
        # 각 row마다 메모리 정리
        cleanup_memory()
    
    return eval_df


def save_test_results(eval_df, output_dir: str, model_name: str, raw_label_field: str):
    """
    평가 결과를 CSV 파일로 저장합니다.
    
    Args:
        eval_df (pd.DataFrame): 저장할 평가 결과 데이터프레임.
        output_dir (str): 출력 디렉토리.
        model_name (str): 모델 이름.
        raw_label_field (str): 실제 레이블 컬럼명.
    
    Returns:
        float: 테스트 정확도
    """
    output_path = Path(output_dir) / "test_results"
    output_path.mkdir(exist_ok=True)
    
    # 정확도 계산
    if 'Pred' in eval_df.columns and raw_label_field in eval_df.columns:
        correct = (eval_df['Pred'] == eval_df[raw_label_field]).sum()
        total = len(eval_df)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n=== 테스트 결과 ===")
        print(f"전체 샘플 수: {total}")
        print(f"정확도: {accuracy:.4f} ({correct}/{total})")
        
        # MLflow에 메트릭 로깅
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_total_samples", total)
        mlflow.log_metric("test_correct_predictions", correct)
        
        # 상세한 분류 리포트
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(
                eval_df[raw_label_field], 
                eval_df['Pred'], 
                target_names=['Normal', 'Abnormal(Nonspecific)', 'Abnormal(Interictal)', 'Abnormal(Ictal)'],
                output_dict=True
            )
            
            # 각 클래스별 정확도 로깅
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    mlflow.log_metric(f"test_{class_name}_precision", metrics['precision'])
                    mlflow.log_metric(f"test_{class_name}_recall", metrics['recall'])
                    mlflow.log_metric(f"test_{class_name}_f1", metrics['f1-score'])
            
            print(f"\n상세 분류 리포트:\n{classification_report(eval_df[raw_label_field], eval_df['Pred'], target_names=['Normal', 'Abnormal(Nonspecific)', 'Abnormal(Interictal)', 'Abnormal(Ictal)'])}")
            
            # 혼동 행렬 로깅
            cm = confusion_matrix(eval_df[raw_label_field], eval_df['Pred'], labels=['Normal', 'Abnormal(Nonspecific)', 'Abnormal(Interictal)', 'Abnormal(Ictal)'])
            cm_file = output_path / "confusion_matrix.txt"
            with open(cm_file, 'w') as f:
                f.write(str(cm))
            mlflow.log_artifact(str(cm_file), "confusion_matrix")
            
        except Exception as e:
            print(f"분류 리포트 생성 중 오류: {e}")
    
    # 결과 저장
    result_file = output_path / f"{model_name}_test_results.csv"
    eval_df.to_csv(result_file, index=False)
    
    # MLflow에 결과 파일 아티팩트로 로깅
    mlflow.log_artifact(str(result_file), "test_results")
    
    print(f"테스트 결과가 {result_file}에 저장되었습니다.")
    
    return accuracy


def build_lora_config_for_seq_cls() -> LoraConfig:
    """분류 태스크용 기본 LoRA 설정을 생성합니다."""

    return LoraConfig(
        lora_alpha=3,
        lora_dropout=0.1,
        r=4,
        bias="none",
        task_type="SEQ_CLS",
    )


def run_hf_training(
    model_conf: ModelConfig, 
    data_conf: DataConfig, 
    trainer_conf: TrainerConfig,
    mlflow_run=None
) -> None:
    """HF/TRL 조합으로 파인튜닝을 수행하고 학습 완료 후 테스트 데이터로 평가합니다."""

    # flash-attn 여부 및 dtype 결정
    _, torch_dtype = maybe_install_flash_attn()

    # 양자화 설정
    quant_config = build_quant_config(model_conf.load_in_4bit, torch_dtype)

    # 모델/토크나이저
    model, tokenizer = load_model_and_tokenizer(
        model_conf.model_id, model_conf.model_dict, num_labels=model_conf.num_labels, quant_config=quant_config
    )

    # 데이터 로드/전처리
    train_dataset, eval_dataset = load_and_prepare_datasets(
        str(data_conf.train_csv_path),
        str(data_conf.eval_csv_path),
        tokenizer,
        text_field=data_conf.text_field,
        raw_label_field=data_conf.raw_label_field,
        max_length=data_conf.max_seq_length,
    )

    # LoRA 설정
    peft_params = build_lora_config_for_seq_cls()

    # 학습 인자
    training_args = TrainingArguments(
        output_dir=str(trainer_conf.output_dir),
        num_train_epochs=trainer_conf.num_train_epochs,
        per_device_train_batch_size=trainer_conf.per_device_train_batch_size,
        per_device_eval_batch_size=trainer_conf.per_device_eval_batch_size,
        gradient_accumulation_steps=trainer_conf.gradient_accumulation_steps,
        learning_rate=trainer_conf.learning_rate,
        weight_decay=trainer_conf.weight_decay,
        fp16=trainer_conf.fp16,
        bf16=trainer_conf.bf16,
        optim=trainer_conf.optim,
        save_steps=trainer_conf.save_steps,
        logging_steps=trainer_conf.logging_steps,
        max_grad_norm=trainer_conf.max_grad_norm,
        max_steps=trainer_conf.max_steps,
        warmup_ratio=trainer_conf.warmup_ratio,
        group_by_length=trainer_conf.group_by_length,
        lr_scheduler_type=trainer_conf.lr_scheduler_type,
        report_to="none",  # MLflow와 충돌 방지
        gradient_checkpointing=trainer_conf.gradient_checkpointing,
        evaluation_strategy=trainer_conf.eval_strategy,
        eval_steps=trainer_conf.eval_steps,
        load_best_model_at_end=trainer_conf.load_best_model_at_end,
    )

    # 트레이너 구성 및 학습
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_params,
        args=training_args,
        dataset_text_field=data_conf.text_field,
        packing=True,
        max_seq_length=data_conf.max_seq_length,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    print("🚀 학습을 시작합니다...")
    
    # 학습 중 메트릭 수집을 위한 콜백
    class MLflowCallback:
        def __init__(self):
            self.step = 0
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value, step=self.step)
                self.step += 1
    
    # MLflow 콜백 추가
    trainer.add_callback(MLflowCallback())
    
    # 학습 실행
    train_result = trainer.train()
    
    print("✅ 학습이 완료되었습니다!")
    
    # 학습 결과 MLflow에 로깅
    if train_result:
        mlflow.log_metric("train_loss", train_result.training_loss)
        mlflow.log_metric("train_runtime", train_result.metrics.get("train_runtime", 0))
        mlflow.log_metric("train_samples_per_second", train_result.metrics.get("train_samples_per_second", 0))
    
    # 모델 저장
    model_save_path = Path(trainer_conf.output_dir) / "final_model"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    
    # MLflow에 모델 아티팩트 로깅
    mlflow.log_artifact(str(model_save_path), "model")
    
    # 학습 완료 후 테스트 데이터로 평가 (data_config에서 직접 사용)
    if data_conf.test_csv_path and data_conf.test_csv_path.exists():
        print("\n🧪 학습 완료된 모델로 테스트 데이터를 평가합니다...")
        
        try:
            # 테스트 데이터 로드
            test_df = pd.read_csv(data_conf.test_csv_path)
            
            # 컬럼명 확인 및 통일
            if data_conf.text_field not in test_df.columns:
                print(f"경고: '{data_conf.text_field}' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {test_df.columns.tolist()}")
                return
            
            # 레이블 컬럼이 있는지 확인 (data_config의 raw_label_field 사용)
            if data_conf.raw_label_field not in test_df.columns:
                print(f"경고: '{data_conf.raw_label_field}' 컬럼을 찾을 수 없습니다. 테스트 평가를 건너뜁니다.")
                return
            
            # 모델 평가
            test_df_with_predictions = evaluate_model_on_test_data(
                test_df, model, tokenizer, data_conf.text_field
            )
            
            # 결과 저장
            model_name = Path(trainer_conf.output_dir).name
            test_accuracy = save_test_results(test_df_with_predictions, trainer_conf.output_dir, model_name, data_conf.raw_label_field)
            
            # 최종 성능 요약
            print(f"\n🎯 최종 테스트 정확도: {test_accuracy:.4f}")
            
        except Exception as e:
            print(f"테스트 평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 메모리 정리
            cleanup_memory()
    
    elif data_conf.test_csv_path:
        print(f"경고: 테스트 데이터 파일을 찾을 수 없습니다: {data_conf.test_csv_path}")
    
    else:
        print("테스트 데이터 경로가 제공되지 않아 테스트 평가를 건너뜁니다.")
    
    print("�� 모든 작업이 완료되었습니다!")

