# 🚀 Git 컨벤션 가이드 (2024 최신)

## 📋 개요
의료 AI 프로젝트를 위한 최신 Git 컨벤션 가이드입니다. 리포지토리 명명법, 커밋 메시지 규칙, 브랜치 전략을 포함합니다.

---

## 🏷️ 리포지토리 명명법

### **기본 규칙**
- **소문자** 사용
- **하이픈(-)** 으로 단어 구분
- **의미있는 이름** 사용
- **언어 표기** 포함 (선택사항)

### **의료 AI 프로젝트 명명 예시**

#### **✅ 좋은 예시**
```
attention-mil-service          # Attention MIL 암 진단 서비스
gastro-endo-classifier        # 위장 내시경 분류기
liver-segmentation-api        # 간 세분화 API
cycle-gan-medical             # 의료 영상 CycleGAN
medical-ai-portfolio          # 의료 AI 포트폴리오
```

#### **❌ 피해야 할 예시**
```
AttentionMILService           # 대문자 사용
attention_mil_service         # 언더스코어 사용
project1                      # 의미없는 이름
medical-ai-project-2024       # 날짜 포함 (불필요)
```

### **프로젝트별 명명 패턴**

#### **1. 서비스형 프로젝트**
```
[기술명]-[기능]-service
예: attention-mil-service, gastro-endo-service
```

#### **2. API형 프로젝트**
```
[기능]-api
예: liver-segmentation-api, medical-diagnosis-api
```

#### **3. 연구형 프로젝트**
```
[기술명]-[도메인]
예: cycle-gan-medical, attention-mil-cancer
```

#### **4. 포트폴리오**
```
[도메인]-[목적]
예: medical-ai-portfolio, ai-research-portfolio
```

---

## 📝 커밋 메시지 규칙

### **Conventional Commits 표준**

#### **기본 형식**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### **Type 종류**
- **feat**: 새로운 기능 추가
- **fix**: 버그 수정
- **docs**: 문서 수정
- **style**: 코드 포맷팅 (기능 변경 없음)
- **refactor**: 코드 리팩토링
- **test**: 테스트 추가/수정
- **chore**: 빌드 프로세스 또는 보조 도구 변경

#### **의료 AI 특화 Type**
- **model**: AI 모델 관련 변경
- **api**: API 엔드포인트 변경
- **mlops**: MLOps 도구 변경
- **medical**: 의료 특화 기능 변경
- **deploy**: 배포 관련 변경

### **커밋 메시지 예시**

#### **✅ 좋은 예시**
```bash
# 기능 추가
feat(api): add cancer diagnosis endpoint
feat(model): implement attention MIL architecture
feat(mlops): add MLflow model registry

# 버그 수정
fix(preprocessing): resolve image normalization issue
fix(api): handle empty image upload error

# 문서 수정
docs: update API documentation
docs(medical): add clinical validation guide

# 리팩토링
refactor(model): separate feature extractor class
refactor(api): restructure response format

# 의료 특화
medical: add HIPAA compliance logging
medical: implement DICOM file support

# 배포
deploy: add Docker containerization
deploy: configure production environment
```

#### **❌ 피해야 할 예시**
```bash
# 너무 짧거나 모호한 메시지
fix bug
update
add stuff

# 과도하게 긴 메시지
feat: implement comprehensive attention mechanism with multiple instance learning for cancer diagnosis in medical imaging with real-time processing capabilities

# 일관성 없는 형식
FIX: bug in preprocessing
Add new feature
UPDATE DOCS
```

### **커밋 메시지 작성 팁**

#### **1. 제목 작성**
- **50자 이내**로 작성
- **명령형** 사용 (add, fix, update)
- **구체적**이고 **명확한** 설명

#### **2. 본문 작성 (필요시)**
```bash
feat(api): add cancer diagnosis endpoint

- Implement POST /api/v1/predict endpoint
- Add image preprocessing pipeline
- Include confidence score calculation
- Add input validation for medical images

Closes #123
```

#### **3. Footer 작성**
```bash
feat(model): implement attention MIL

BREAKING CHANGE: Model input format changed from single image to image sequence
Fixes #456
Relates to #789
```

---

## 🌿 브랜치 전략

### **Git Flow 기반 전략**

#### **메인 브랜치**
- **main**: 프로덕션 배포용
- **develop**: 개발 통합용

#### **보조 브랜치**
- **feature/**: 새로운 기능 개발
- **hotfix/**: 긴급 버그 수정
- **release/**: 릴리스 준비

### **브랜치 명명법**

#### **Feature 브랜치**
```
feature/[type]/[description]
예: feature/api/cancer-diagnosis-endpoint
예: feature/model/attention-mil-implementation
예: feature/mlops/mlflow-integration
```

#### **Hotfix 브랜치**
```
hotfix/[description]
예: hotfix/api-image-upload-error
예: hotfix/model-loading-issue
```

#### **Release 브랜치**
```
release/[version]
예: release/v1.0.0
예: release/v1.1.0
```

### **브랜치 사용 예시**

#### **새로운 기능 개발**
```bash
# 1. develop 브랜치에서 feature 브랜치 생성
git checkout develop
git pull origin develop
git checkout -b feature/api/cancer-diagnosis-endpoint

# 2. 개발 및 커밋
git add .
git commit -m "feat(api): add cancer diagnosis endpoint"

# 3. develop 브랜치로 병합
git checkout develop
git merge feature/api/cancer-diagnosis-endpoint
git push origin develop

# 4. feature 브랜치 삭제
git branch -d feature/api/cancer-diagnosis-endpoint
```

#### **릴리스 준비**
```bash
# 1. release 브랜치 생성
git checkout develop
git checkout -b release/v1.0.0

# 2. 버전 정보 업데이트
git commit -m "chore: bump version to 1.0.0"

# 3. main과 develop에 병합
git checkout main
git merge release/v1.0.0
git tag v1.0.0

git checkout develop
git merge release/v1.0.0

# 4. release 브랜치 삭제
git branch -d release/v1.0.0
```

---

## 🏥 의료 AI 특화 규칙

### **보안 관련 커밋**
```bash
# 개인정보 보호
security: add patient data encryption
security: implement HIPAA compliance measures

# 감사 로그
audit: add medical data access logging
audit: implement audit trail for predictions
```

### **의료 데이터 관련**
```bash
# 데이터 처리
data: add DICOM file support
data: implement medical image preprocessing
data: add patient data anonymization

# 모델 성능
performance: improve cancer detection accuracy
performance: optimize inference speed for real-time diagnosis
```

### **임상 검증**
```bash
# 임상적 검증
clinical: add medical expert validation
clinical: implement clinical accuracy metrics
clinical: add medical guideline compliance
```

---

## 📋 프로젝트별 적용 예시

### **Attention MIL 프로젝트**
```bash
# 리포지토리명
attention-mil-service

# 주요 커밋 메시지
feat(model): implement attention MIL architecture
feat(api): add cancer diagnosis endpoint
feat(mlops): add MLflow model registry
medical: add clinical validation metrics
deploy: add Docker containerization
```

### **GastroEndo 프로젝트**
```bash
# 리포지토리명
gastro-endo-service

# 주요 커밋 메시지
feat(model): implement transfer learning with EfficientNet
feat(api): add endoscopy location classification
feat(medical): add 7-class location mapping
deploy: configure production deployment
```

### **Liver Segmentation 프로젝트**
```bash
# 리포지토리명
liver-segmentation-api

# 주요 커밋 메시지
feat(model): implement 3D U-Net architecture
feat(api): add liver segmentation endpoint
feat(medical): add IoU and Dice coefficient metrics
deploy: add GPU support for 3D processing
```

---

## 🛠️ Git Hooks 설정

### **커밋 메시지 검증**
```bash
# .git/hooks/commit-msg
#!/bin/sh
commit_regex='^(feat|fix|docs|style|refactor|test|chore|model|api|mlops|medical|deploy)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "❌ 커밋 메시지 형식이 올바르지 않습니다."
    echo "✅ 형식: type(scope): description"
    echo "✅ 예시: feat(api): add cancer diagnosis endpoint"
    exit 1
fi
```

### **브랜치명 검증**
```bash
# .git/hooks/pre-commit
#!/bin/sh
branch_regex='^(main|develop|feature|hotfix|release)/.+'

if ! echo "$(git branch --show-current)" | grep -qE "$branch_regex"; then
    echo "❌ 브랜치명 형식이 올바르지 않습니다."
    echo "✅ 형식: type/description"
    echo "✅ 예시: feature/api/cancer-diagnosis"
    exit 1
fi
```

---

## 📊 커밋 히스토리 예시

### **완성된 프로젝트의 커밋 히스토리**
```bash
* feat(api): add cancer diagnosis endpoint
* feat(model): implement attention MIL architecture
* feat(mlops): add MLflow model registry
* feat(medical): add clinical validation metrics
* feat(preprocessing): add medical image preprocessing
* feat(deploy): add Docker containerization
* docs: add comprehensive API documentation
* test: add unit tests for model inference
* style: format code according to PEP 8
* chore: update dependencies to latest versions
* fix(api): handle empty image upload error
* feat(security): add HIPAA compliance logging
* deploy: configure production environment
* docs(medical): add clinical validation guide
```

---

## 🎯 실무 적용 체크리스트

### **프로젝트 시작 시**
- [ ] 리포지토리명 결정 (kebab-case)
- [ ] 브랜치 전략 설정 (Git Flow)
- [ ] Git hooks 설정
- [ ] .gitignore 파일 구성

### **개발 중**
- [ ] feature 브랜치 사용
- [ ] Conventional Commits 형식 준수
- [ ] 의미있는 커밋 메시지 작성
- [ ] 정기적인 develop 브랜치 병합

### **릴리스 시**
- [ ] release 브랜치 생성
- [ ] 버전 태그 생성
- [ ] main 브랜치 병합
- [ ] 릴리스 노트 작성

---

이 가이드를 따라하면 **일관성 있고 전문적인 Git 히스토리**를 만들 수 있으며, **팀 협업**과 **프로젝트 관리**가 훨씬 수월해집니다! 🚀 