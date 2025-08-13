"""
Unsloth 기반 파인튜닝 트레이너

- Unsloth 라이브러리를 활용한 효율적인 파인튜닝
- 기존 config 카테고리와 호환되는 구조
- 프롬프트 기반 분류 태스크 지원
- MLflow 통합 및 테스트 데이터 자동 평가
- 실시간 모니터링을 통한 수동 Early Stopping
"""

from __future__ import annotations

import torch
import pandas as pd
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import time
import signal
import sys
from sklearn.metrics import classification_report, confusion_matrix

from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
import mlflow
import mlflow.pytorch
from datasets import Dataset

from .config import ModelConfig, DataConfig, TrainerConfig


# ============================================================================
# 전역 변수 및 유틸리티 함수
# ============================================================================

# 전역 변수로 Early Stopping 플래그 관리
early_stop_requested = False

def cleanup_memory():
    """메모리 정리 함수 (GPU/CPU 모두 적용)"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()


# ============================================================================
# 프롬프트 생성 및 텍스트 처리 함수
# ============================================================================

def create_train_prompt(text: str, label: str) -> str:
    """EEG 분류를 위한 훈련용 프롬프트 생성."""
    
    instruction = "Analyze the given EEG report and classify the status based on the following categories:"
    
    prompt = f"""### Instruction:
{instruction}

### EEG Report:
{text}

### Classification:
{label}<|endoftext|>"""
    
    return prompt

def create_test_prompt(text: str) -> str:
    """테스트용 프롬프트 생성 (레이블 없이)"""
    
    instruction = "Analyze the given EEG report and classify the status based on the following categories:"
    
    prompt = f"""### Instruction:
{instruction}

### EEG Report:
{text}

### Classification:"""
    
    return prompt

def extract_classification_label(predicted_text: str) -> str:
    """모델이 생성한 텍스트에서 분류 레이블을 추출합니다."""
    
    try:
        # 가능한 레이블들
        LABELS = ['Abnormal(Nonspecific)', 'Abnormal(Interictal)', 'Abnormal(Ictal)', 'Normal']
        
        # "### Classification:" 이후의 텍스트에서 레이블 찾기
        if "### Classification:" in predicted_text:
            classification_part = predicted_text.split("### Classification:")[-1].strip()
            
            
            for label in LABELS:
                if label.lower() in classification_part.lower():
                    print(f"Debug - 레이블 추출: {label}")
                    return label
        
        # 전체 텍스트에서 레이블 찾기 (백업)
        for label in LABELS:
            if label.lower() in predicted_text.lower():
                print(f"Debug - 전체 텍스트에서 레이블 추출: {label}")
                return label
        
        print(f"⚠️ 레이블을 찾을 수 없습니다. 전체 텍스트: {predicted_text}")
        return 'Unknown'
        
    except Exception as e:
        print(f"❌ 레이블 추출 오류: {e}")
        return 'Unknown'


# ============================================================================
# 데이터셋 생성 및 전처리 함수
# ============================================================================

def create_text_dataset(data_config: DataConfig, tokenizer, split="train"):
    """
    SFTTrainer에 적합한 'text' 필드를 포함한 데이터셋을 생성합니다.
    """
    if split == "train":
        csv_path = data_config.train_csv_path
    elif split == "eval":
        csv_path = data_config.eval_csv_path
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # pandas로 데이터 로드
    df = pd.read_csv(csv_path)
    
    # 텍스트 포맷팅 함수
    def _format_row_to_text(row):
        text = create_train_prompt(
            row[data_config.text_field],
            row[data_config.raw_label_field]
        )
        return {'text': text}

    # 데이터프레임을 Dataset으로 변환하고 'text' 필드를 추가
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(_format_row_to_text, remove_columns=df.columns.tolist())
    
    return dataset


# ============================================================================
# 모델 예측 및 평가 함수
# ============================================================================

def predict_label(pred):
    """
    예측된 레이블 인덱스를 실제 레이블로 변환합니다.
    
    Args:
        pred (torch.Tensor): 예측된 레이블 인덱스.
    
    Returns:
        str: 실제 레이블.
    """
    # 상수 정의
    LABELS = ['Abnormal(Nonspecific)', 'Abnormal(Interictal)', 'Abnormal(Ictal)', 'Normal']
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
    try:
        # 디바이스 확인 및 설정
        device = next(model.parameters()).device
        
        # 테스트용 프롬프트 생성(without label)
        test_prompt = create_test_prompt(text)

        # 토크나이징
        inputs = tokenizer(
            test_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024
        )
        
        # 입력을 모델과 같은 디바이스로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 모델을 평가 모드로 설정
        model.eval()

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            top_p=0.9,
            temperature=0.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # 토큰 ID를 텍스트로 디코딩
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f'Debug: {predicted_text}')

        return extract_classification_label(predicted_text)
        
    except Exception as e:
        print(f"❌ test_model 오류: {e}")
        import traceback
        traceback.print_exc()
        raise e

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
    
    # 디바이스 확인
    device = next(model.parameters()).device

    # Pred 컬럼 초기화
    if 'Pred' not in eval_df.columns:
        eval_df['Pred'] = None

    # 'Pred' 컬럼의 dtype을 명시적으로 object로 변환하여 FutureWarning 방지
    if eval_df['Pred'].dtype != object:
        eval_df['Pred'] = eval_df['Pred'].astype(object)

    for idx, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="Evaluating"):
        try:
            predicted_label = test_model(model, tokenizer, row[text_field])
            eval_df.at[idx, 'Pred'] = predicted_label
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
        
        # MLflow에 기본 메트릭 로깅
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_total_samples", total)
        mlflow.log_metric("test_correct_predictions", correct)
        mlflow.log_metric("test_incorrect_predictions", total - correct)
        mlflow.log_metric("test_error_rate", 1 - accuracy)
        
        # 상세한 분류 리포트
        try:
            report = classification_report(
                eval_df[raw_label_field], 
                eval_df['Pred'], 
                target_names=['Abnormal(Nonspecific)', 'Abnormal(Interictal)', 'Abnormal(Ictal)', 'Normal'],
                output_dict=True
            )
            
            # 각 클래스별 정확도 로깅
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    mlflow.log_metric(f"test_{class_name}_precision", metrics['precision'])
                    mlflow.log_metric(f"test_{class_name}_recall", metrics['recall'])
                    mlflow.log_metric(f"test_{class_name}_f1", metrics['f1-score'])
                    mlflow.log_metric(f"test_{class_name}_support", metrics['support'])
            
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


# ============================================================================
# MLflow 콜백 클래스
# ============================================================================

class MLflowCallback(TrainerCallback):
    """MLflow 메트릭 로깅을 위한 콜백 클래스 (자동 Early Stopping 및 최고 성능 모델 저장 포함)"""
    
    def __init__(self, patience=10, min_improvement=0.01, min_delta=0.001, model_save_dir=None):
        self.step = 0
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        self.should_stop = False
        
        # 최고 성능 모델 저장을 위한 변수들
        self.model_save_dir = model_save_dir
        self.best_model_path = None
        self.best_step = 0
        self.model = None
        self.tokenizer = None
        
        print(f"🚀 MLflow 콜백 초기화: patience={patience}, min_improvement={min_improvement}, min_delta={min_delta}")
        if model_save_dir:
            print(f"💾 최고 성능 모델 자동 저장이 활성화되었습니다: {model_save_dir}")
    
    def set_model_and_tokenizer(self, model, tokenizer):
        """모델과 토크나이저를 설정합니다."""
        self.model = model
        self.tokenizer = tokenizer
    
    def on_train_begin(self, args, state, control, **kwargs):
        """학습 시작 시 호출됩니다."""
        print("🚀 MLflow 콜백이 활성화되었습니다.")
        print("🔄 자동 Early Stopping 모니터링이 활성화되었습니다.")
        if self.model_save_dir:
            print("💾 최고 성능 모델 자동 저장이 활성화되었습니다.")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 이벤트 시 호출됩니다."""
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=self.step)
                    
                    # 손실 모니터링 및 자동 Early Stopping 체크
                    if key == "loss":
                        self.loss_history.append(value)
                        self._check_early_stopping(value)
                        
                        # 최고 성능 모델 저장 체크
                        if self.model_save_dir and self.model is not None:
                            self._check_and_save_best_model(value, state.global_step)
                        
                        # Early Stopping 조건 만족 시 학습 중단 신호
                        if self.should_stop:
                            print(f"🛑 Early Stopping 조건 만족! Step {self.step}에서 학습을 중단합니다.")
                            control.should_training_stop = True
            
            self.step += 1
    
    def _check_early_stopping(self, current_loss):
        """Early Stopping 조건을 체크합니다."""
        # 첫 번째 손실값은 건너뛰기
        if len(self.loss_history) <= 1:
            return
        
        # 이전 손실값과 비교
        previous_loss = self.loss_history[-2]
        
        # 개선 여부 확인 (최소 변화량 고려)
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            print(f"✅ 새로운 최고 성능! Loss: {self.best_loss:.4f} (Step: {self.step})")
        else:
            self.patience_counter += 1
            print(f"⚠️ 개선 없음 ({self.patience_counter}/{self.patience}): Loss {current_loss:.4f} vs Best {self.best_loss:.4f}")
            
            # Early Stopping 조건 확인
            if self.patience_counter >= self.patience:
                print(f"🛑 {self.patience}번 연속 개선 없음. Early Stopping 조건 만족!")
                self.should_stop = True
    
    def _check_and_save_best_model(self, current_loss, global_step):
        """최고 성능 모델을 저장합니다."""
        if current_loss < self.best_loss - self.min_delta:
            try:
                # 이전 최고 모델 삭제
                if self.best_model_path and self.best_model_path.exists():
                    import shutil
                    shutil.rmtree(self.best_model_path)
                    print(f"🗑️ 이전 최고 모델 삭제: {self.best_model_path}")
                
                # 새로운 최고 모델 저장
                self.best_model_path = Path(self.model_save_dir) / f"best_model_step_{global_step}_loss_{current_loss:.4f}"
                self.best_model_path.mkdir(parents=True, exist_ok=True)
                
                # 모델과 토크나이저 저장
                self.model.save_pretrained(str(self.best_model_path))
                self.tokenizer.save_pretrained(str(self.best_model_path))
                
                # 메타데이터 저장
                metadata = {
                    "step": global_step,
                    "loss": current_loss,
                    "timestamp": time.time(),
                    "model_type": "unsloth_best"
                }
                
                import json
                with open(self.best_model_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                self.best_step = global_step
                print(f"💾 새로운 최고 성능 모델 저장: {self.best_model_path}")
                print(f"   Step: {global_step}, Loss: {current_loss:.4f}")
                
                # MLflow에 아티팩트로 로깅
                mlflow.log_artifact(str(self.best_model_path), f"best_model_step_{global_step}")
                
            except Exception as e:
                print(f"❌ 최고 성능 모델 저장 중 오류: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """학습 종료 시 호출됩니다."""
        print("✅ MLflow 콜백이 완료되었습니다.")
        if self.should_stop:
            print(f"🛑 Early Stopping으로 인해 학습이 중단되었습니다.")
        else:
            print(f"📊 정상적으로 학습이 완료되었습니다.")
        print(f"📈 최종 손실: {self.best_loss:.4f}")
        print(f"📊 손실 히스토리: {len(self.loss_history)} 스텝")
        
        if self.best_model_path and self.best_model_path.exists():
            print(f"🏆 최고 성능 모델: {self.best_model_path}")
            print(f"   Step: {self.best_step}, Loss: {self.best_loss:.4f}")
    
    def get_best_model_path(self):
        """최고 성능 모델 경로를 반환합니다."""
        return self.best_model_path


# ============================================================================
# 메인 실행 함수들
# ============================================================================

def run_unsloth_training(
    model_conf: ModelConfig, 
    data_conf: DataConfig, 
    trainer_conf: TrainerConfig,
    mlflow_run=None
) -> None:
    """Unsloth 기반 분류 파인튜닝을 수행하고 학습 완료 후 테스트 데이터로 평가합니다."""

    global early_stop_requested
    
    print("🚀 Unsloth 기반 파인튜닝을 시작합니다...")
    print("🔄 자동 Early Stopping이 활성화되었습니다.")
    
    # 1. 모델과 토크나이저 로드 (4비트 양자화)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_conf.model_dict[model_conf.model_id],
        max_seq_length=data_conf.max_seq_length,
        dtype=None,
        load_in_4bit=model_conf.load_in_4bit,
        full_finetuning=False
    )
    
    # 2. QLoRA 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA 랭크
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 메모리 절약
        random_state=3407,
    )
    
    # 3. 데이터셋 로드 및 전처리 (input_ids 포함)
    print("📊 데이터셋을 로드하고 전처리합니다...")
    train_dataset = create_text_dataset(data_conf, tokenizer, "train")
    eval_dataset = create_text_dataset(data_conf, tokenizer, "eval")
    
    # 데이터셋 검증
    print("🔍 데이터셋 형식을 검증합니다...")
    print(f"훈련 데이터셋 샘플 수: {len(train_dataset)}")
    print(f"검증 데이터셋 샘플 수: {len(eval_dataset)}")
    print(f"첫 번째 샘플 키: {list(train_dataset[0].keys())}")
    
    # 4. 학습 인자 설정 (Unsloth 호환 - Early Stopping 옵션 제거)
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
        optim="adamw_8bit",  # Unsloth 권장
        save_steps=trainer_conf.save_steps,
        logging_steps=trainer_conf.logging_steps,
        max_grad_norm=trainer_conf.max_grad_norm,
        max_steps=trainer_conf.max_steps,
        warmup_ratio=trainer_conf.warmup_ratio,
        group_by_length=trainer_conf.group_by_length,
        lr_scheduler_type=trainer_conf.lr_scheduler_type,
        report_to="none",  # MLflow와 충돌 방지
        gradient_checkpointing=True,  # Unsloth와 호환
        seed=3407,
    )
    
    # 5. SFTTrainer 구성 및 학습
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=data_conf.max_seq_length,
        args=training_args,
        packing=True,  # 시퀀스 패킹으로 효율성 향상
        packing_efficiency=0.95,
    )
    
    print("🚀 학습을 시작합니다...")
    print("💡 MLflow UI에서 실시간으로 손실을 모니터링하세요:")
    print(f"   mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlruns.db")
    print("🔄 자동 Early Stopping이 활성화되어 손실이 수렴하면 자동으로 중단됩니다.")
    
    # MLflow 콜백 추가 (자동 Early Stopping)
    mlflow_callback = MLflowCallback(patience=20, min_improvement=0.005, min_delta=0.0005, model_save_dir=trainer_conf.output_dir)
    trainer.add_callback(mlflow_callback)
    mlflow_callback.set_model_and_tokenizer(model, tokenizer) # 모델과 토크나이저 설정
    
    try:
        # 학습 실행
        train_result = trainer.train()
        
        if early_stop_requested:
            print("🛑 사용자 요청으로 학습이 중단되었습니다.")
        elif mlflow_callback.should_stop:
            print("🛑 자동 Early Stopping으로 학습이 중단되었습니다.")
        else:
            print("✅ 학습이 완료되었습니다!")
        
        # 학습 결과 MLflow에 로깅
        if train_result:
            mlflow.log_metric("train_loss", train_result.training_loss)
            mlflow.log_metric("train_runtime", train_result.metrics.get("train_runtime", 0))
            mlflow.log_metric("train_samples_per_second", train_result.metrics.get("train_samples_per_second", 0))
            mlflow.log_metric("training_completed", 1 if not early_stop_requested and not mlflow_callback.should_stop else 0)
            mlflow.log_metric("early_stopping_triggered", 1 if mlflow_callback.should_stop else 0)
        
    except KeyboardInterrupt:
        print("\n🛑 키보드 인터럽트로 학습이 중단되었습니다.")
        early_stop_requested = True
        mlflow.log_metric("training_interrupted", 1)
    
    # 6. 모델 저장
    output_path = Path(trainer_conf.output_dir) / "unsloth_model"
    model.save_pretrained(str(output_path), push_to_hub=False)
    tokenizer.save_pretrained(str(output_path), push_to_hub=False)
    
    # MLflow에 모델 아티팩트 로깅
    mlflow.log_artifact(str(output_path), "model")
    
    # 최고 성능 모델 정보 표시
    best_model_path = mlflow_callback.get_best_model_path()
    if best_model_path and best_model_path.exists():
        print(f"🏆 최고 성능 모델 정보:")
        print(f"   경로: {best_model_path}")
        print(f"   Step: {mlflow_callback.best_step}")
        print(f"   Loss: {mlflow_callback.best_loss:.4f}")
        
        # 최고 성능 모델을 최종 모델로 복사 (선택사항)
        import shutil
        final_best_path = Path(trainer_conf.output_dir) / "best_model_final"
        if final_best_path.exists():
            shutil.rmtree(final_best_path)
        shutil.copytree(best_model_path, final_best_path)
        print(f"💾 최고 성능 모델이 {final_best_path}에 최종 복사되었습니다.")
        
        # MLflow에 최종 최고 성능 모델 로깅
        mlflow.log_artifact(str(final_best_path), "best_model_final")
    
    if early_stop_requested or mlflow_callback.should_stop:
        print(f"🛑 중단된 Unsloth 모델이 {output_path}에 저장되었습니다.")
    else:
        print(f"✅ 완료된 Unsloth 모델이 {output_path}에 저장되었습니다.")
    
    print("✅ 학습 단계가 완료되었습니다.")

def run_unsloth_testing(
    model_conf: ModelConfig,
    data_conf: DataConfig,
    trainer_conf: TrainerConfig,
    model_path: Path = None,
    mlflow_run=None
) -> float:
    """학습 완료된 Unsloth 모델로 테스트 데이터를 평가합니다."""
    
    print("🧪 Unsloth 모델 테스트를 시작합니다...")
    
    # 최고 성능 모델 우선 탐색
    best_model_path = None
    if model_path:
        # best_model_final 디렉토리 확인
        final_best_path = model_path / "best_model_final"
        if final_best_path.exists():
            best_model_path = final_best_path
            print(f"🏆 최고 성능 모델을 찾았습니다: {best_model_path}")
        else:
            # 기존 unsloth_model 디렉토리 사용
            best_model_path = model_path / "unsloth_model"
            print(f"📥 기본 모델을 사용합니다: {best_model_path}")
    
    if not best_model_path or not best_model_path.exists():
        print(f"❌ 모델 경로를 찾을 수 없습니다: {best_model_path}")
        return 0.0
    
    try:
        # 1. 저장된 모델과 토크나이저 로드
        print(f"📥 저장된 모델을 로드합니다: {best_model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(best_model_path),
            max_seq_length=data_conf.max_seq_length,
            dtype=None,
            load_in_4bit=model_conf.load_in_4bit,
            full_finetuning=False
        )
        
        # 메타데이터 확인 (최고 성능 모델인 경우)
        metadata_path = best_model_path / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"📊 모델 메타데이터:")
            print(f"   Step: {metadata.get('step', 'N/A')}")
            print(f"   Loss: {metadata.get('loss', 'N/A')}")
            print(f"   타입: {metadata.get('model_type', 'N/A')}")
        
        # 2. 테스트 데이터 평가
        if data_conf.test_csv_path and data_conf.test_csv_path.exists():
            print("📊 테스트 데이터를 로드하고 평가합니다...")
            
            # 테스트 데이터 로드
            test_df = pd.read_csv(data_conf.test_csv_path)
            
            # 컬럼명 확인 및 통일
            if data_conf.text_field not in test_df.columns:
                print(f"경고: '{data_conf.text_field}' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {test_df.columns.tolist()}")
                return 0.0
            
            # 레이블 컬럼이 있는지 확인
            if data_conf.raw_label_field not in test_df.columns:
                print(f"경고: '{data_conf.raw_label_field}' 컬럼을 찾을 수 없습니다. 테스트 평가를 건너뜁니다.")
                return 0.0
            
            # 모델 평가
            test_df_with_predictions = evaluate_model_on_test_data(
                test_df, model, tokenizer, data_conf.text_field
            )
            
            # 결과 저장
            model_name = Path(trainer_conf.output_dir).name
            test_accuracy = save_test_results(test_df_with_predictions, trainer_conf.output_dir, model_name, data_conf.raw_label_field)
            
            # 최종 성능 요약
            print(f"\n🎯 최종 테스트 정확도: {test_accuracy:.4f}")
            
            return test_accuracy
            
        elif data_conf.test_csv_path:
            print(f"경고: 테스트 데이터 파일을 찾을 수 없습니다: {data_conf.test_csv_path}")
            return 0.0
        
        else:
            print("테스트 데이터 경로가 제공되지 않아 테스트 평가를 건너뜁니다.")
            return 0.0
            
    except Exception as e:
        print(f"❌ 테스트 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 0.0
    
    finally:
        # 메모리 정리
        cleanup_memory()

def run_unsloth_train_testing(
    model_conf: ModelConfig, 
    data_conf: DataConfig, 
    trainer_conf: TrainerConfig,
    mlflow_run=None
) -> float:
    """
    Unsloth 기반 파인튜닝을 수행하고 학습 완료 후 테스트 데이터로 평가하는 통합 함수
    
    Args:
        model_conf: 모델 설정
        data_conf: 데이터 설정  
        trainer_conf: 트레이너 설정
        mlflow_run: MLflow 실행 컨텍스트 (선택사항)
    
    Returns:
        float: 테스트 정확도
    """
    
    print("🚀 Unsloth 파인튜닝 및 테스트를 시작합니다...")
    print("=" * 60)
    
    try:
        # 1단계: 모델 파인튜닝 실행
        print("📚 1단계: 모델 파인튜닝을 시작합니다...")
        run_unsloth_training(
            model_conf=model_conf,
            data_conf=data_conf, 
            trainer_conf=trainer_conf,
            mlflow_run=mlflow_run
        )
        
        print("✅ 파인튜닝이 완료되었습니다.")
        print("=" * 60)
        
        # 2단계: 학습된 모델로 테스트 실행
        print("🧪 2단계: 학습된 모델로 테스트를 시작합니다...")
        model_path = Path(trainer_conf.output_dir)
        
        test_accuracy = run_unsloth_testing(
            model_conf=model_conf,
            data_conf=data_conf,
            trainer_conf=trainer_conf,
            model_path=model_path,
            mlflow_run=mlflow_run
        )
        
        print("✅ 테스트가 완료되었습니다.")
        print("=" * 60)
        
        # 최종 결과 요약
        print("🎯 최종 결과 요약")
        print(f"📊 테스트 정확도: {test_accuracy:.4f}")
        print(f"💾 모델 저장 경로: {model_path}")
        print(f"📁 결과 파일 경로: {model_path / 'test_results'}")
        
        # 최고 성능 모델 정보 표시
        best_model_final_path = model_path / "best_model_final"
        if best_model_final_path.exists():
            print(f"🏆 최고 성능 모델 경로: {best_model_final_path}")
            # 메타데이터 읽기
            metadata_path = best_model_final_path / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"   Step: {metadata.get('step', 'N/A')}")
                print(f"   Loss: {metadata.get('loss', 'N/A')}")
        
        # MLflow에 최종 메트릭 로깅
        if mlflow_run:
            mlflow.log_metric("final_test_accuracy", test_accuracy)
            mlflow.log_metric("training_and_testing_completed", 1)
            mlflow.set_tag("pipeline_status", "completed")
        
        return test_accuracy
        
    except Exception as e:
        print(f"❌ 파인튜닝 및 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # MLflow에 오류 로깅
        if mlflow_run:
            mlflow.log_metric("training_and_testing_failed", 1)
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("error_message", str(e))
        
        # 메모리 정리
        cleanup_memory()
        raise e
    
    finally:
        # 최종 메모리 정리
        cleanup_memory()
        print("🧹 메모리 정리가 완료되었습니다.")