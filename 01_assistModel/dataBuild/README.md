# PPT 접근성 진단 자동화 — 핵심 README

## Pipeline

![진단 대상 화면 선정 파이프라인](./assets/dataBuild.png)

## 목적
PPTX 접근성 진단 보고서에서 페이지별 메타데이터를 전사하고, 
오류 영역과 코드가 왜 그렇게 평가되었는지를 설명하는 추론을 자동 생성하여 metadata.jsonl로 저장합니다.

## 필수 사항
- Python 3.9+
- Google Chrome (또는 Chromium)
- 패키지: `python-pptx`, `selenium`, `webdriver-manager`, `openai`

```bash
pip install python-pptx selenium webdriver-manager openai
```

## 환경 변수
- `OPENAI_API_KEY`를 환경 변수로 설정(권장: `.env`).

## 사용법
```python
from main import process_pptx
process_pptx(
    pptx_path=r"C:\path\to\sample.pptx",
    output_jsonl="metadata.jsonl",
    debug=True,
    headless=True,
)
```

## 동작 개요
1. **PPT 파싱**: 슬라이드의 텍스트/표/그룹 내부까지 재귀 수집.
2. **AI 전사**: 아래 스키마에 맞춰 보이는 그대로 전사(누락 시 빈 문자열).
3. **스크린샷**: URL 유효 시 전체 페이지 풀캡처(`image/전체영역/`).
4. **오류 이미지 추출**: 슬라이드 내 그림(blob) 저장(`image/오류영역/`).
5. **근거 생성**: 평가 입력(오류 영역 및 코드)에 대한 접근성 진단을 진행했을때, 왜 이런 평가 출력(검사항목, 오류유형, 문제점 및 개선방안_텍스트, 문제점 및 개선방안_코드)이 나왔는지에 대한 근거(rationale)를 작성.
6. **출력**: 슬라이드별 1라인 JSON을 `metadata.jsonl`에 기록.

## 전사 스키마(요약)
```json
{
  "URL": "string",
  "검사항목": "string",
  "오류유형": "string",
  "문제점": "string",                    // 오류 영역 코드
  "문제점 및 개선방안_텍스트": "string",  
  "문제점 및 개선방안_코드": "string"   
}
```

## 출력 형식(JSONL 예시)
> 참고: 아래 형식은 **Hugging Face Datasets**의 *Image dataset* 가이드를 참고했습니다.
> 문서: https://huggingface.co/docs/datasets/en/image_dataset
```json
{"file_names": ["./image/전체영역/1.png", "./image/오류영역/1.png"],
 "output": {"추론": "...", "검사항목": "...", "오류유형": "...",
            "문제점": "...", "문제점 및 개선방안_텍스트": "...",
            "문제점 및 개선방안_코드": "..."}}
```

## 디렉터리 구조
```
project-root/
├─ image/
│  ├─ 전체영역/        # 슬라이드별 전체 페이지 스크린샷
│  └─ 오류영역/        # 슬라이드 내 오류 영역 이미지(복수 가능)
├─ metadata.jsonl       # 최종 결과(슬라이드별 1라인 JSON)
├─ main.py              # (예) 실행 스크립트
└─ requirements.txt
```


