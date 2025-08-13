#!/usr/bin/env python3
"""
CDE-EEG-LLM 파인튜닝 메인 스크립트

사용법:
    python scripts/train.py --method hf          # Hugging Face 기반
    python scripts/train.py --method unsloth     # Unsloth 기반
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

from src.training.config import ModelConfig, DataConfig, TrainerConfig, MLflowConfig
from src.training.trainer_hf import run_hf_training
from src.training.trainer_unsloth import run_unsloth_training, run_unsloth_testing, run_unsloth_train_testing

def force_environment_variables():
    """환경 변수를 강제로 설정합니다."""
    
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # 캐시 디렉토리 자동 생성
    (cache_dir / "unsloth").mkdir(exist_ok=True)
    (cache_dir / "huggingface").mkdir(exist_ok=True)
    (cache_dir / "transformers").mkdir(exist_ok=True)
    (cache_dir / "datasets").mkdir(exist_ok=True)
    (cache_dir / "unsloth" / "compiled").mkdir(parents=True, exist_ok=True)
    (cache_dir / "unsloth" / "models").mkdir(parents=True, exist_ok=True)
    
    # 환경 변수 강제 설정
    os.environ["UNSLOTH_CACHE_DIR"] = str(cache_dir / "unsloth")
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["UNSLOTH_COMPILE_CACHE_DIR"] = str(cache_dir / "unsloth" / "compiled")
    os.environ["UNSLOTH_MODEL_CACHE_DIR"] = str(cache_dir / "unsloth" / "models")
    
    print(" 환경 변수 강제 설정 완료:")
    for key, value in os.environ.items():
        if "CACHE" in key or "HF_" in key or "UNSLOTH" in key:
            print(f"   {key}: {value}")

def create_default_configs(method:str) -> tuple[ModelConfig, DataConfig, TrainerConfig, MLflowConfig]:
    """기본 설정값을 생성합니다."""
    
    # 모델 설정
    model_config = ModelConfig(
        model_id="gpt-oss-20b",
        model_dict={
            'Llama-3.1-8B': 'meta-llama/Llama-3.1-8B',
            'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
            'Llama-3.1-70B': 'meta-llama/Llama-3.1-70B',
            'Llama-3.2-3B': 'meta-llama/Llama-3.2-3B',
            'Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
            'falcon-7B': 'tiiuae/falcon-7b',
            'falcon-11B': 'tiiuae/falcon-11B',
            't5-small': 'google-t5/t5-small',
            'gpt-oss-20b': 'unsloth/gpt-oss-20b',
        },
        num_labels=4,
        use_qlora=True,
        load_in_4bit=True,
        use_flash_attn_if_available=True,
    )
    
    # 데이터 설정
    data_config = DataConfig(
        train_csv_path=Path("data/250723_train.csv"),
        eval_csv_path=Path("data/250723_eval.csv"),
        test_csv_path=Path("data/250723_test.csv"),
        text_field="EMR",
        raw_label_field="True",
        max_seq_length=1024,
    )
    
    # MLflow 설정
    # DB 파일을 mlruns 디렉토리 내부에 위치하도록 경로 지정
    mlflow_config = MLflowConfig(
        experiment_name="cde-eeg-llm-finetuning",
        tracking_uri="mlruns",  # 파일 시스템 경로로 MLflow 트래킹
        artifact_location="mlruns",
        log_artifacts=True,
        log_models=True,
        log_hyperparams=True,
        log_metrics=True,
    )
    
    # 학습 설정
    trainer_config = TrainerConfig(
        output_dir=Path(f"output/{model_config.model_id}_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        save_steps=25,
        logging_steps=10,
        max_grad_norm=1.0,
        max_steps=-1,
        warmup_ratio=0.1,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="mlflow",
        gradient_checkpointing=True,
    )
    
    return model_config, data_config, trainer_config, mlflow_config


def setup_mlflow_experiment(mlflow_config: MLflowConfig, model_id: str, method: str):
    """MLflow 실험을 설정하고 시작합니다."""
    
    # MLflow 설정
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    # 실험 이름 설정
    experiment_name = f"{mlflow_config.experiment_name}_{method.upper()}"
    mlflow.set_experiment(experiment_name)
    
    # 실험 시작
    with mlflow.start_run(run_name=f"{model_id}_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"🔬 MLflow 실험 시작: {experiment_name}")
        print(f"📊 Run ID: {run.info.run_id}")
        print(f"🔗 MLflow UI: mlflow ui --backend-store-uri {mlflow_config.tracking_uri}")
        
        return run


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
        "--config",
        type=str,
        help="설정 파일 경로 (선택사항)"
    )
    
    args = parser.parse_args()
    
    # 🚨 환경 변수를 가장 먼저 설정
    force_environment_variables()

    # 기본 설정 로드
    model_config, data_config, trainer_config, mlflow_config = create_default_configs(args.method)
    
    print(f"🚀 CDE-EEG-LLM 파인튜닝 시작")
    print(f"📋 방법: {args.method.upper()}")
    print(f"🧠 모델: {model_config.model_id}")
    print(f"📊 훈련 데이터: {data_config.train_csv_path}")
    print(f"📊 검증 데이터: {data_config.eval_csv_path}")
    print(f"🧪 테스트 데이터: {data_config.test_csv_path}")
    print(f"💾 출력: {trainer_config.output_dir}")
    print(f"🔬 MLflow 실험: {mlflow_config.experiment_name}")
    print("-" * 50)
    
    # 출력 디렉토리 생성
    trainer_config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow 실험 설정
    mlflow_run = setup_mlflow_experiment(mlflow_config, model_config.model_id, args.method)
    
    try:
        # MLflow에 하이퍼파라미터 로깅
        mlflow.log_params({
            "model_id": model_config.model_id,
            "method": args.method,
            "num_labels": model_config.num_labels,
            "use_qlora": model_config.use_qlora,
            "load_in_4bit": model_config.load_in_4bit,
            "max_seq_length": data_config.max_seq_length,
            "num_train_epochs": trainer_config.num_train_epochs,
            "learning_rate": trainer_config.learning_rate,
            "batch_size": trainer_config.per_device_train_batch_size,
            "gradient_accumulation_steps": trainer_config.gradient_accumulation_steps,
        })
        
        # 데이터셋 정보 로깅
        mlflow.log_params({
            "train_data": str(data_config.train_csv_path),
            "eval_data": str(data_config.eval_csv_path),
            "test_data": str(data_config.test_csv_path),
            "text_field": data_config.text_field,
            "label_field": data_config.raw_label_field,
        })
        
        # 캐시 정보 로깅
        mlflow.log_params({
            "unsloth_cache_dir": os.environ.get('UNSLOTH_CACHE_DIR'),
            "hf_home": os.environ.get('HF_HOME'),
            "transformers_cache": os.environ.get('TRANSFORMERS_CACHE'),
            "datasets_cache": os.environ.get('HF_DATASETS_CACHE'),
        })
        
        if args.method == "hf":
            print("🔄 Hugging Face 기반 파인튜닝 시작...")
            run_hf_training(model_config, data_config, trainer_config, mlflow_run)
        elif args.method == "unsloth":
            print("⚡ Unsloth 기반 파인튜닝 시작...")
            run_unsloth_train_testing(model_config, data_config, trainer_config, mlflow_run)
        
        print("✅ 파인튜닝 완료!")
        
        # 최종 모델 등록
        if mlflow_config.log_models:
            model_path = trainer_config.output_dir / "final_model"
            if model_path.exists():
                mlflow.pytorch.log_model(
                    pytorch_model=model_path,
                    artifact_path="model",
                    registered_model_name=f"cde-eeg-llm-{args.method}-{model_config.model_id}"
                )
                print(f"🏷️ 모델이 MLflow에 등록되었습니다: cde-eeg-llm-{args.method}-{model_config.model_id}")
        # 성공 상태 로깅
        mlflow.log_param("status", "completed")

    except Exception as e:
        print(f"❌ 파인튜닝 중 오류 발생: {e}")
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
        sys.exit(1)
    
    finally:
        print(f"🔬 MLflow 실험 완료: {mlflow_run.info.run_id}")


if __name__ == "__main__":
    main()

