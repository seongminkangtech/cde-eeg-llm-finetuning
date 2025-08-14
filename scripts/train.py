#!/usr/bin/env python3
"""
CDE-EEG-LLM 파인튜닝 메인 스크립트

사용법:
    python scripts/train.py --method hf          # Hugging Face 기반
    python scripts/train.py --method unsloth     # Unsloth 기반
    python scripts/train.py --config configs/training.yaml  # 설정 파일 사용
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import os
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.config import TrainingConfig
from src.training.trainer_hf import run_hf_training
from src.training.trainer_unsloth import run_unsloth_training, run_unsloth_testing, run_unsloth_train_testing

def setup_environment(training_config: TrainingConfig):
    """환경 변수를 설정하고 캐시 디렉토리를 생성합니다."""
    
    # 캐시 디렉토리 생성
    training_config.cache.create_cache_directories()

    # 환경 변수 설정
    env_vars = training_config.cache.get_environment_variables()
    for key, value in env_vars.items():
        os.environ[key] = value

    print("🔧 환경 변수 설정 완료:")
    for key, value in env_vars.items():
        print(f"   {key}: {value}")


def setup_mlflow_experiment(training_config: TrainingConfig, method: str):
    """MLflow 실험을 설정하고 시작합니다."""
    
    # MLflow 설정
    mlflow.set_tracking_uri(training_config.mlflow.tracking_uri)
    
    # 실험 이름 설정
    experiment_name = f"{training_config.mlflow.experiment_name}_{method.upper()}"
    mlflow.set_experiment(experiment_name)
    
def log_training_params(training_config: TrainingConfig, method: str):
    """MLflow에 학습 파라미터들을 로깅합니다."""
    
    # 모델 관련 파라미터
    mlflow.log_params({
        "model_id": training_config.model.model_id,
        "method": method,
        "num_labels": training_config.model.num_labels,
        "use_qlora": training_config.model.use_qlora,
        "load_in_4bit": training_config.model.load_in_4bit,
        "max_seq_length": training_config.data.max_seq_length,
        "num_train_epochs": training_config.trainer.num_train_epochs,
        "learning_rate": training_config.trainer.learning_rate,
        "batch_size": training_config.trainer.per_device_train_batch_size,
        "gradient_accumulation_steps": training_config.trainer.gradient_accumulation_steps,
    })
    
    # 데이터셋 정보
    mlflow.log_params({
        "train_data": str(training_config.data.train_csv_path),
        "eval_data": str(training_config.data.eval_csv_path),
        "test_data": str(training_config.data.test_csv_path),
        "text_field": training_config.data.text_field,
        "label_field": training_config.data.raw_label_field,
    })
    
    # 캐시 정보
    mlflow.log_params({
        "unsloth_cache_dir": str(training_config.cache.unsloth_cache_dir),
        "hf_home": str(training_config.cache.huggingface_cache_dir),
        "transformers_cache": str(training_config.cache.transformers_cache_dir),
        "datasets_cache": str(training_config.cache.datasets_cache_dir),
    })


def load_training_config(config_path: Path, model_id: str, method: str, project_root: Path) -> TrainingConfig:
    """설정을 로드합니다. 설정 파일이 제공된 경우 YAML에서 로드하고, 그렇지 않으면 기본값을 사용합니다."""
    
    if config_path and config_path.exists():
        print(f"📋 설정 파일에서 설정 로드: {config_path}")
        return TrainingConfig.from_yaml(config_path, model_id, method, project_root)
    else:
        print("📋 기본 설정 사용")
        return TrainingConfig.create_default_config(model_id, method, project_root)


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(description="CDE-EEG-LLM 파인튜닝")
    parser.add_argument(
        "--method",
        choices=["hf", "unsloth"],
        default="unsloth",
        help="파인튜닝 방법 선택 (기본값: unsloth)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-oss-20b",
        help="사용할 모델 ID (기본값: gpt-oss-20b)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="설정 파일 경로 (선택사항)"
    )
    
    args = parser.parse_args()
    
    # 🚨 프로젝트 루트 설정
    project_root = Path(__file__).parent.parent
    
    # 🚨 설정 파일 경로 처리
    config_path = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
    
    # 🚨 통합 설정 생성
    training_config = load_training_config(config_path, args.model_id, args.method, project_root)
    
    # 🚨 환경 변수를 가장 먼저 설정
    setup_environment(training_config)
    
    # 출력 디렉토리 동적 설정
    training_config.trainer.output_dir = Path(f"output/{training_config.model.model_id}_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    print(f"🚀 CDE-EEG-LLM 파인튜닝 시작")
    print(f"📋 방법: {args.method.upper()}")
    print(f"🧠 모델: {training_config.model.model_id}")
    print(f"📊 훈련 데이터: {training_config.data.train_csv_path}")
    print(f"📊 검증 데이터: {training_config.data.eval_csv_path}")
    print(f"🧪 테스트 데이터: {training_config.data.test_csv_path}")
    print(f"💾 출력: {training_config.trainer.output_dir}")
    print(f"🔬 MLflow 실험: {training_config.mlflow.experiment_name}")
    print("-" * 50)
    
    # 출력 디렉토리 생성
    training_config.trainer.output_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow 실험 설정
    setup_mlflow_experiment(training_config, args.method)
    
    try:
        # MLflow에 파라미터 로깅
        log_training_params(training_config, args.method)
        
        if args.method == "hf":
            print("🔄 Hugging Face 기반 파인튜닝 시작...")
            run_hf_training(training_config.model, training_config.data, training_config.trainer)
        elif args.method == "unsloth":
            print("⚡ Unsloth 기반 파인튜닝 시작...")
            run_unsloth_train_testing(training_config.model, training_config.data, training_config.trainer)
        
        print("✅ 파인튜닝 완료!")
        
        # 최종 모델 등록 - 실제 저장 경로 사용
        if training_config.mlflow.log_models:
            # 🎯 실제 저장 경로 사용
            if args.method == "unsloth":
                model_path = training_config.trainer.output_dir / "final_model"
            else:
                model_path = training_config.trainer.output_dir / "final_model"
                
            if model_path.exists():
                mlflow.pytorch.log_model(
                    pytorch_model=model_path,
                    artifact_path="model",
                    registered_model_name=f"cde-eeg-llm-{args.method}-{training_config.model.model_id}"
                )
                print(f"🏷️ 모델이 MLflow에 등록되었습니다: cde-eeg-llm-{args.method}-{training_config.model.model_id}")
            else:
                print(f"⚠️ 모델 경로를 찾을 수 없습니다: {model_path}")
        
        mlflow.log_param("status", "completed",)

    except Exception as e:
        print(f"❌ 파인튜닝 중 오류 발생: {e}")
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
        sys.exit(1)
    
    finally:
        print(f"🔬 MLflow 실험 완료")


if __name__ == "__main__":
    main()

