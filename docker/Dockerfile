# CDE-EEG-LLM-Finetuning 프로젝트 Docker 이미지
FROM nvcr.io/nvidia/pytorch:25.01-py3

# 기본 환경 변수 설정
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 캐시 디렉토리 환경변수 추가 (cache_dir = cache)
ENV UNSLOTH_CACHE_DIR=/workspace/cde-eeg-llm-finetuning/cache/unsloth
ENV HF_HOME=/workspace/cde-eeg-llm-finetuning/cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/cde-eeg-llm-finetuning/cache/transformers
ENV HF_DATASETS_CACHE=/workspace/cde-eeg-llm-finetuning/cache/datasets

# 캐시 디렉토리 미리 생성 (추가)
RUN mkdir -p ${UNSLOTH_CACHE_DIR} \
           ${HF_HOME} \
           ${TRANSFORMERS_CACHE} \
           ${HF_DATASETS_CACHE}

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt