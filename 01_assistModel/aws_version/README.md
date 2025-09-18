
# 🧭 Bedrock_SFT_VLM.ipynb — 코드 구조 & 동작 설명(리포지토리용 README)

이 문서는 **`Bedrock_SFT_VLM.ipynb`** 노트북 **그 자체의 구현 내용**을 기준으로 작성되었습니다.  
노트북은 **Hugging Face 데이터 로딩 → 이미지 S3 업로드 → Bedrock Conversation JSONL 변환 → Bedrock SFT(커스터마이징) → 프로비저닝(Provisioned Throughput) → 테스트 호출 → SageMaker 파이프라인 실행/모니터링**까지 한 흐름으로 구성됩니다.

> 리전/계정별 지원 모델, 권한, API 스펙은 달라질 수 있습니다. 실제 실행 전 계정/리전의 Bedrock 문서를 확인하세요.

---

## 📦 사전 준비(코드에 등장하는 요구사항)
- **AWS Region**: 기본값 `us-west-2` (코드 주석대로 *Llama 3.2* 파인튜닝 가용 기준)
- **필수 권한/IAM**: S3, Bedrock(모델 커스터마이징/런타임), STS, SageMaker Pipeline, CloudWatch Logs
- **환경/라이브러리(노트북 상단 설치 셀)**  
  `boto3`, `datasets`, `Pillow`, `tqdm`, `langchain` (+ SageMaker SDK 구성 요소)
- **실행용 상수**
  - `ROLE_ARN`: 파이프라인/스텝 실행에 사용할 IAM Role ARN
  - `INST_TYPE`: 파이프라인 스텝 인스턴스 타입(예: `ml.m5.xlarge`)

---

## 🗂 데이터 & 입력
- **소스 데이터셋**: `doodoo77/For_VLM_accessibility` (Hugging Face `datasets.load_dataset`로 로드)
- **이미지 업로드 규칙**  
  각 `subset`(예: `train`)의 이미지를 **PNG**로 변환 후  
  `s3://{bucket_name}/images/{subset}/{i:03d}.png` 경로로 업로드
- **버킷명 생성**  
  STS에서 `account_id`를 얻어 **고유 버킷명** 구성:  
  `vlm-accessibility-{account_id}-{region}` (존재 시 재사용)

---

## 🔧 전처리/아티팩트 생성 — 코드가 하는 일

### 1) 이미지 → S3 업로드 (`upload_images_to_s3`)
SageMaker `@step` 데코레이터로 파이프라인 스텝화.  
- `datasets.load_dataset("doodoo77/For_VLM_accessibility")`에서 각 예시의 이미지를 **PIL → PNG 바이트**로 변환
- `S3Uploader.upload_bytes`로 S3에 개별 업로드
- 리턴값: `s3://{bucket}/images/{subset}` (해당 subset의 이미지 폴더 URI)

### 2) Bedrock Conversation JSONL 변환 (`convert_to_bedrock_jsonl`, `write_jsonl_file`)
노트북 내 **Bedrock 대화 포맷**으로 한 레코드씩 변환 후 JSONL 저장:
```json
{
  "schemaVersion": "bedrock-conversation-2024",
  "system": [{ "text": "<system_prompt 내용>" }],
  "messages": [
    {
      "role": "user",
      "content": [
        { "text": "<user_prompt(html)>" },
        {
          "image": {
            "format": "png",
            "source": { "s3Location": { "uri": "s3://.../images/<subset>/<id>.png", "bucketOwner": "<account_id>" } }
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [ { "text": "<assistant_prompts(outputs)>" } ]
    }
  ]
}
```
- `write_jsonl_file(subset, image_folder)`는 데이터셋의 각 예시에 대해 위 구조를 **1줄 JSON**로 기록 (`{subset}.jsonl`)
- 이미지 S3 URI는 위 1) 단계의 경로를 사용

### 3) JSONL → S3 업로드 (`upload_jsonl_to_s3` 스텝)
- `{subset}.jsonl`을 `s3://{bucket}/data/{subset}.jsonl`로 업로드
- 파이프라인에서 다음 단계의 학습 입력으로 사용

---

## 🧠 프롬프트(코드 내 정의)
- `system_prompt`: 대화 규약/사고-해결 형식 안내 (노트북에 명시된 멘트가 그대로 삽입됨)
- `user_prompt = PromptTemplate(input_variables=["html"], template="...")`  
  → **접근성 평가** 지시가 핵심. HTML 스니펫을 입력 변수로 받아 설명을 요구
- `assistant_prompts(outputs)`: 데이터의 `outputs`(각 항목의 `rationale`, `eval_items`, `eval_reason`, `plan`)을 템플릿(`assisstant_prompt`)에 채워 **모범 응답 텍스트 블록**을 생성

> 즉, **입력은** “이미지 + 관련 HTML 스니펫 + 지시문”, **출력은** “접근성 평가에 대한 설명(라셔널)과 항목별 근거/개선안 텍스트”입니다.  
> 이 구조가 SFT 학습의 **instruction ↔ ideal answer** 페어로 사용됩니다.

---

## 🏋️ 학습(SFT) — `train` 스텝
- Bedrock `create_model_customization_job` 호출로 **FINE_TUNING** 잡 생성
- **입력 데이터**: `s3://{bucket}/data/{subset}.jsonl` (2)단계에서 만든 JSONL
- **하이퍼파라미터(예시)**:  
  `epochCount=2`, `learningRate`, `batchSize` 등(코드에 정의)
- **베이스 모델**: 파라미터 `base_model_id`로 전달 (예: `meta.llama3-2-11b-instruct-v1:0` — 노트북 하단에서 설정)
- **결과**: 커스터마이즈된 모델 ARN/ID 반환 및 잡 상태 폴링

> 코드 상 코멘트엔 Titan 예시 문구가 있으나, 실제 실행 파라미터는 **Vision Llama 3.2 계열로 설정**되어 있습니다.

---

## ⚙️ 프로비저닝 — `create_prov_thruput` 스텝
- `create_provisioned_model_throughput` 로 **Provisioned Throughput** 생성
- 상태가 `Creating` → 완료될 때까지 루프 폴링

---

## 🔮 테스트 호출 — `test_model` 스텝
- Bedrock Runtime `invoke_model`로 **멀티모달 요청** 전송
- 페이로드에는
  - `instruction`/`text` 지시,
  - `images`: (base64/포맷) 리스트(코드에서 생성)  
  를 포함
- 응답에서 `generation` 필드 추출/출력

> 테스트는 **학습된 커스텀 모델의 응답 형태**를 확인하는 용도입니다.

---

## 🚀 파이프라인 구성/실행
- 파라미터 설정:
  - `pipeline_name = "VLM-fine-tune-pipeline"`
  - `custom_model_name`, `training_job_name`, `base_model_id`, `provisioned_model_name` 등 타임스탬프 포함
- **스텝 연결 순서**(코드 그대로):
  1) `image_folder = upload_images_to_s3("train")`  
  2) `upload_jsonl_to_s3_step_result = upload_jsonl_to_s3(image_folder, "train")`  
  3) `model_id = train(..., upload_jsonl_to_s3_step_result)`  
  4) `create_prov_thruput_response = create_prov_thruput(model_id, provisioned_model_name)`  
  5) `test_model_response = test_model(create_prov_thruput_response)`  
- `Pipeline(...).upsert(role_arn)` → `start()` → `describe()`  
- 마지막 셀에는 `pipeline_execution_complete` **Waiter** 예제가 포함(파이프라인 ARN 수동 입력)

---

## 📋 빠른 실행 체크(코드 기반)
- [ ] `ROLE_ARN`, `INST_TYPE`, `region`(`us-west-2`) 확인
- [ ] STS로 `account_id` 정상 조회 → 버킷명 생성/사용
- [ ] 이미지 업로드 경로: `s3://{bucket}/images/{subset}/{i:03d}.png`
- [ ] JSONL 업로드 경로: `s3://{bucket}/data/{subset}.jsonl`
- [ ] `base_model_id`(예: `meta.llama3-2-11b-instruct-v1:0`) 및 SFT 하이퍼파라미터 확인
- [ ] Provisioned Throughput 생성 및 상태 확인
- [ ] 테스트 스텝 응답(`generation`) 출력 확인
- [ ] 파이프라인 Waiter에 **실제 ARN** 반영

---

## 🛡️ 주의/운영 팁
- **권한 오류**(AccessDenied 등) → IAM 정책/Role 신뢰 정책 확인
- **이미지 권한** → S3 접근/버킷 소유자 ID(`bucketOwner`) 설정 확인
- **모델/포맷 불일치** → Bedrock 모델별 멀티모달 스키마 차이 주의
- **리전 제약** → Vision SFT/Provisioned Throughput 지원 리전 확인
- **비용** → SFT 잡/프로비저닝/추론 호출 모두 비용 발생

---

## 📎 참고(코드 내 대표 심볼)
- 상수: `ROLE_ARN`, `INST_TYPE`, `region`, `bucket_name`, `account_id`
- 전처리: `upload_images_to_s3`, `write_jsonl_file`, `upload_jsonl_to_s3`
- 포맷 변환: `convert_to_bedrock_jsonl` (Bedrock Conversation 2024 + S3 이미지 소스)
- 학습: `train`(create_model_customization_job)
- 프로비저닝: `create_prov_thruput`
- 테스트: `test_model`(invoke_model)
- 파이프라인: `Pipeline(name=..., steps=[...])`, `upsert`, `start`, `describe`

---

본 README는 **노트북 내부 코드 내용**을 바탕으로 작성되었습니다. 저장소에 맞게 Role/모델 ID/버킷/데이터셋 등을 프로젝트 환경에 맞춰 업데이트해 주세요.
