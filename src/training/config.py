"""
파인튜닝 설정 모듈

- 본 모듈은 파인튜닝에 필요한 설정 값을 데이터클래스로 정의합니다.
- 각 설정은 명시적으로 타입을 지정하여 가독성과 안전성을 높입니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import yaml


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
    model_dict: Dict[str, str] = field(default_factory=lambda: {
        'Llama-3.1-8B': 'meta-llama/Llama-3.1-8B',
        'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
        'Llama-3.1-70B': 'meta-llama/Llama-3.1-70B',
        'Llama-3.2-3B': 'meta-llama/Llama-3.2-3B',
        'Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
        'falcon-7B': 'tiiuae/falcon-7b',
        'falcon-11B': 'tiiuae/falcon-11B',
        't5-small': 'google-t5/t5-small',
        'gpt-oss-20b': 'unsloth/gpt-oss-20b',
    })
    num_labels: int = 4
    use_qlora: bool = True
    load_in_4bit: bool = True
    use_flash_attn_if_available: bool = True


@dataclass
class DataConfig:
    """데이터 관련 설정 값."""

    train_csv_path: Path = Path("data/250723_train.csv")
    eval_csv_path: Path = Path("data/250723_eval.csv")
    test_csv_path: Path = Path("data/250723_test.csv")
    text_field: str = "EMR"
    raw_label_field: str = "True"
    max_seq_length: int = 1024


@dataclass
class MLflowConfig:
    """MLflow 관련 설정 값."""
    
    experiment_name: str = "cde-eeg-llm-finetuning"
    tracking_uri: str = "mlruns"  # 파일 시스템 경로로 MLflow 트래킹
    artifact_location: str = "mlruns"
    log_artifacts: bool = True
    log_models: bool = True
    log_hyperparams: bool = True
    log_metrics: bool = True


@dataclass
class TrainerConfig:
    """트레이너/학습 관련 설정 값."""

    output_dir: Path = None  # 동적으로 설정됨
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    fp16: bool = False
    bf16: bool = True
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


@dataclass
class CacheConfig:
    """캐시 디렉토리 및 환경 변수 설정."""
    
    project_root: Path = None  # 동적으로 설정됨
    cache_dir: Path = None  # 동적으로 설정됨
    
    def __post_init__(self):
        """프로젝트 루트와 캐시 디렉토리를 자동으로 설정합니다."""
        if self.project_root is None:
            # 현재 파일에서 프로젝트 루트 추정
            current_file = Path(__file__)
            self.project_root = current_file.parent.parent.parent
        
        if self.cache_dir is None:
            self.cache_dir = self.project_root / "cache"
    
    @property
    def unsloth_cache_dir(self) -> Path:
        return self.cache_dir / "unsloth"
    
    @property
    def huggingface_cache_dir(self) -> Path:
        return self.cache_dir / "huggingface"
    
    @property
    def transformers_cache_dir(self) -> Path:
        return self.cache_dir / "transformers"
    
    @property
    def datasets_cache_dir(self) -> Path:
        return self.cache_dir / "datasets"
    
    @property
    def unsloth_compiled_cache_dir(self) -> Path:
        return self.cache_dir / "unsloth" / "compiled"
    
    @property
    def unsloth_model_cache_dir(self) -> Path:
        return self.cache_dir / "unsloth" / "models"
    
    def create_cache_directories(self):
        """필요한 캐시 디렉토리들을 생성합니다."""
        self.cache_dir.mkdir(exist_ok=True)
        self.unsloth_cache_dir.mkdir(exist_ok=True)
        self.huggingface_cache_dir.mkdir(exist_ok=True)
        self.transformers_cache_dir.mkdir(exist_ok=True)
        self.datasets_cache_dir.mkdir(exist_ok=True)
        self.unsloth_compiled_cache_dir.mkdir(parents=True, exist_ok=True)
        self.unsloth_model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_environment_variables(self) -> Dict[str, str]:
        """환경 변수 설정을 위한 딕셔너리를 반환합니다."""
        return {
            "UNSLOTH_CACHE_DIR": str(self.unsloth_cache_dir),
            "HF_HOME": str(self.huggingface_cache_dir),
            "TRANSFORMERS_CACHE": str(self.transformers_cache_dir),
            "HF_DATASETS_CACHE": str(self.datasets_cache_dir),
            "UNSLOTH_COMPILE_CACHE_DIR": str(self.unsloth_compiled_cache_dir),
            "UNSLOTH_MODEL_CACHE_DIR": str(self.unsloth_model_cache_dir),
        }


@dataclass
class TrainingConfig:
    """전체 파인튜닝 설정을 통합 관리하는 클래스."""
    
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    mlflow: MLflowConfig
    cache: CacheConfig
    
    @classmethod
    def create_default_config(cls, model_id: str, method: str, project_root: Path = None) -> 'TrainingConfig':
        """기본 설정값을 생성합니다."""
        
        # 캐시 설정
        cache_config = CacheConfig(project_root=project_root)
        
        # 모델 설정
        model_config = ModelConfig(model_id=model_id)
        
        # 데이터 설정
        data_config = DataConfig()
        
        # MLflow 설정
        mlflow_config = MLflowConfig()
        
        # 학습 설정
        trainer_config = TrainerConfig()
        
        return cls(
            model=model_config,
            data=data_config,
            trainer=trainer_config,
            mlflow=mlflow_config,
            cache=cache_config
        )
    
    @classmethod
    def from_yaml(cls, config_path: Path, model_id: str, method: str, project_root: Path = None) -> 'TrainingConfig':
        """YAML 설정 파일에서 설정을 로드합니다."""
        
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 기본 설정 생성
        training_config = cls.create_default_config(model_id, method, project_root)
        
        # YAML에서 설정 오버라이드
        if 'model' in config_data:
            for key, value in config_data['model'].items():
                if hasattr(training_config.model, key):
                    setattr(training_config.model, key, value)
        
        if 'data' in config_data:
            for key, value in config_data['data'].items():
                if hasattr(training_config.data, key):
                    if key.endswith('_path'):
                        setattr(training_config.data, key, Path(value))
                    else:
                        setattr(training_config.data, key, value)
        
        if 'trainer' in config_data:
            for key, value in config_data['trainer'].items():
                if hasattr(training_config.trainer, key):
                    if key == 'output_dir':
                        continue  # output_dir은 동적으로 설정됨
                    setattr(training_config.trainer, key, value)
        
        if 'mlflow' in config_data:
            for key, value in config_data['mlflow'].items():
                if hasattr(training_config.mlflow, key):
                    setattr(training_config.mlflow, key, value)
        
        return training_config
    
    def to_yaml(self, config_path: Path):
        """현재 설정을 YAML 파일로 저장합니다."""
        
        config_data = {
            'model': {
                'model_id': self.model.model_id,
                'num_labels': self.model.num_labels,
                'use_qlora': self.model.use_qlora,
                'load_in_4bit': self.model.load_in_4bit,
                'use_flash_attn_if_available': self.model.use_flash_attn_if_available,
            },
            'data': {
                'train_csv_path': str(self.data.train_csv_path),
                'eval_csv_path': str(self.data.eval_csv_path),
                'test_csv_path': str(self.data.test_csv_path),
                'text_field': self.data.text_field,
                'raw_label_field': self.data.raw_label_field,
                'max_seq_length': self.data.max_seq_length,
            },
            'trainer': {
                'num_train_epochs': self.trainer.num_train_epochs,
                'per_device_train_batch_size': self.trainer.per_device_train_batch_size,
                'per_device_eval_batch_size': self.trainer.per_device_eval_batch_size,
                'gradient_accumulation_steps': self.trainer.gradient_accumulation_steps,
                'learning_rate': self.trainer.learning_rate,
                'weight_decay': self.trainer.weight_decay,
                'fp16': self.trainer.fp16,
                'bf16': self.trainer.bf16,
                'optim': self.trainer.optim,
                'save_steps': self.trainer.save_steps,
                'logging_steps': self.trainer.logging_steps,
                'max_grad_norm': self.trainer.max_grad_norm,
                'max_steps': self.trainer.max_steps,
                'warmup_ratio': self.trainer.warmup_ratio,
                'group_by_length': self.trainer.group_by_length,
                'lr_scheduler_type': self.trainer.lr_scheduler_type,
                'report_to': self.trainer.report_to,
                'gradient_checkpointing': self.trainer.gradient_checkpointing,
            },
            'mlflow': {
                'experiment_name': self.mlflow.experiment_name,
                'tracking_uri': self.mlflow.tracking_uri,
                'artifact_location': self.mlflow.artifact_location,
                'log_artifacts': self.mlflow.log_artifacts,
                'log_models': self.mlflow.log_models,
                'log_hyperparams': self.mlflow.log_hyperparams,
                'log_metrics': self.mlflow.log_metrics,
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)
