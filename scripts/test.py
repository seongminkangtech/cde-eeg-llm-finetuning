#!/usr/bin/env python3
"""
CDE-EEG-LLM 테스트 전용 스크립트

사용법:
    python scripts/test.py --method unsloth --model_path output/gpt-oss-20b_unsloth_20250101_120000/unsloth_model
    python scripts/test.py --method unsloth --output_dir output/gpt-oss-20b_unsloth_20250101_120000
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
from src.training.trainer_unsloth import run_unsloth_testing


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
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["UNSLOTH_COMPILE_CACHE_DIR"] = str(cache_dir / "unsloth" / "compiled")
    os.environ["UNSLOTH_MODEL_CACHE_DIR"] = str(cache_dir / "unsloth" / "models")
    
    print("✅ 환경 변수 강제 설정 완료:")
    for key, value in os.environ.items():
        if "CACHE" in key or "HF_" in key or "UNSLOTH" in key:
            print(f"   {key}: {value}")


def create_test_configs(method: str, model_path: Path = None, output_dir: Path = None) -> tuple[ModelConfig, DataConfig, TrainerConfig, MLflowConfig]:
    """테스트용 설정값을 생성합니다."""
    
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
    mlflow_config = MLflowConfig(
        experiment_name="cde-eeg-llm-testing",
        tracking_uri="mlruns",
        artifact_location="mlruns",
        log_artifacts=True,
        log_models=True,
        log_hyperparams=True,
        log_metrics=True,
    )
    
    # 출력 디렉토리 설정
    if output_dir is None:
        if model_path:
            # model_path에서 상위 디렉토리를 output_dir로 사용
            output_dir = model_path.parent
        else:
            # 기본 출력 디렉토리
            output_dir = Path(f"output/test_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # 학습 설정 (테스트용으로 간소화)
    trainer_config = TrainerConfig(
        output_dir=output_dir,
        num_train_epochs=0,  # 테스트에서는 사용하지 않음
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0,  # 테스트에서는 사용하지 않음
        weight_decay=0,  # 테스트에서는 사용하지 않음
        fp16=False,
        bf16=True,
        optim="adamw_8bit",
        save_steps=0,  # 테스트에서는 사용하지 않음
        logging_steps=1,
        max_grad_norm=0.3,
        max_steps=0,  # 테스트에서는 사용하지 않음
        warmup_ratio=0,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="mlflow",
        gradient_checkpointing=True,
    )
    
    return model_config, data_config, trainer_config, mlflow_config


def setup_mlflow_experiment(mlflow_config: MLflowConfig, method: str, model_path: Path = None):
    """MLflow 실험을 설정하고 시작합니다."""
    
    # MLflow 설정
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    # 실험 이름 설정
    experiment_name = f"{mlflow_config.experiment_name}_{method.upper()}"
    mlflow.set_experiment(experiment_name)
    
    # run_name 생성
    if model_path:
        model_name = model_path.parent.name
    else:
        model_name = "unknown_model"
    
    run_name = f"test_{model_name}_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 실험 시작
    with mlflow.start_run(run_name=run_name) as run:
        print(f"🔬 MLflow 테스트 실험 시작: {experiment_name}")
        print(f"📊 Run ID: {run.info.run_id}")
        print(f"🔗 MLflow UI: mlflow ui --backend-store-uri {mlflow_config.tracking_uri}")
        
        return run


def main():
    """메인 함수."""
    parser = argparse.ArgumentParser(description="CDE-EEG-LLM 테스트")
    parser.add_argument(
        "--method",
        choices=["unsloth"],
        default="unsloth",
        help="테스트할 모델 방법 선택 (기본값: unsloth)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="테스트할 모델 경로 (예: output/gpt-oss-20b_unsloth_20250101_120000/unsloth_model)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="결과를 저장할 출력 디렉토리 (선택사항)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="설정 파일 경로 (선택사항)"
    )
    
    args = parser.parse_args()
    
    # 🚨 환경 변수를 가장 먼저 설정
    force_environment_variables()

    # 모델 경로 설정
    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ 모델 경로를 찾을 수 없습니다: {model_path}")
            sys.exit(1)
        print(f"📁 모델 경로: {model_path}")
    else:
        print("⚠️ --model_path가 지정되지 않았습니다. 기본 경로를 사용합니다.")
    
    # 출력 디렉토리 설정
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    
    # 기본 설정 로드
    model_config, data_config, trainer_config, mlflow_config = create_test_configs(
        args.method, model_path, output_dir
    )
    
    print(f"🚀 CDE-EEG-LLM 테스트 시작")
    print(f"📋 방법: {args.method.upper()}")
    print(f"🧠 모델: {model_config.model_id}")
    print(f"🧪 테스트 데이터: {data_config.test_csv_path}")
    print(f"💾 출력: {trainer_config.output_dir}")
    print(f"📊 MLflow 실험: {mlflow_config.experiment_name}")
    print("-" * 50)
    
    # 출력 디렉토리 생성
    trainer_config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow 실험 설정
    mlflow_run = setup_mlflow_experiment(mlflow_config, args.method, model_path)
    
    try:
        # MLflow에 테스트 정보 로깅
        mlflow.log_params({
            "test_type": "model_evaluation",
            "method": args.method,
            "model_path": str(model_path) if model_path else "default",
            "num_labels": model_config.num_labels,
            "use_qlora": model_config.use_qlora,
            "load_in_4bit": model_config.load_in_4bit,
            "max_seq_length": data_config.max_seq_length,
        })
        
        # 데이터셋 정보 로깅
        mlflow.log_params({
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
        
        # 테스트 실행
        if args.method == "unsloth":
            print("⚡ Unsloth 모델 테스트 시작...")
            test_accuracy = run_unsloth_testing(
                model_config, 
                data_config, 
                trainer_config, 
                model_path, 
                mlflow_run
            )
            print(f"✅ 테스트 완료! 정확도: {test_accuracy:.4f}")
        
        # 성공 상태 로깅
        mlflow.log_param("status", "completed")
        mlflow.log_metric("final_test_accuracy", test_accuracy)

    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
        sys.exit(1)
    
    finally:
        print(f"🔬 MLflow 테스트 실험 완료: {mlflow_run.info.run_id}")


if __name__ == "__main__":
    main()