# VLM 모델의 접근성·진단 능력 평가 페이지 (Streamlit)

[👉 평가 페이지 바로가기](https://snclab-rl.streamlit.app/)

이 프로젝트는 **지도학습된 AI의 접근성 진단 출력(추론 과정)** 에 대해  
**인간 전문가가 체계적으로 피드백을 제공**할 수 있도록 만든 Streamlit 기반 평가 페이지입니다.

전문가는
- 진단 대상 페이지의 **오류 영역 스크린샷**과 **HTML 코드**, **전문가 메모**를 입력으로 제공하고
- AI가 생성한 **접근성 진단 + 추론(JSON)** 을 검토·수정한 뒤
- 최종 **수정된 추론 + 피드백**을 하나의 샘플로 저장합니다.

이렇게 축적된 JSONL 데이터는 추후 **SFT / 강화학습(RL)** 에 바로 활용할 수 있습니다.

---

## 💻 UI 미리보기

<table>
  <tr>
    <td align="center" width="50%">
      <img src="./assets/RL_결과.png" alt="AI 진단 및 추론 출력 화면" width="100%" /><br/>
      <sub><b>그림 1.</b> AI 진단·추론 출력 화면</sub>
    </td>
    <td align="center" width="50%">
      <img src="./assets/RL_추론.png" alt="전문가 피드백 입력 화면" width="100%" /><br/>
      <sub><b>그림 2.</b> 전문가 피드백 입력 화면</sub>
    </td>
  </tr>
</table>

---

## 🔍 주요 기능

- **멀티모달 입력**
  - 오류 영역 스크린샷 이미지 (선택)
  - 오류 영역 HTML 코드
  - 표준 개선방안 Excel 업로드 (검사항목/오류유형 참고용)
  - 전문가 메모(맥락 설명)

- **외부 LLM/VLM 호출**
  - OpenAI-compatible (`ChatOpenAI`)
  - AWS Bedrock (`ChatBedrock`)
  - Vertex AI 튜닝 엔드포인트(Gemini 2.5 Pro, `google-genai` v1)

- **구조화된 AI 출력(JSON)**
  - `reasoning`: 추론 요약 리스트
  - `checks`: `검사항목`, `오류유형`
  - `fix`: 개선방안 설명 + HTML 코드

- **전문가 피드백 UI**
  - 추론·검사항목·오류유형·개선방안(설명/코드) **직접 편집**
  - 만족도 점수(1–5) + 코멘트 입력
  - 수정본/피드백을 한 번에 JSONL로 저장

- **데이터셋 내보내기**
  - `hf_export/data.jsonl` 자동 append
  - 원본 이미지 `hf_export/images/` 저장
  - Hugging Face Hub dataset으로 바로 Push

---

## ⚡ 빠른 시작

### 요구사항

- Python 3.9+
- 선택: OpenAI API 키, AWS 자격증명, GCP 인증(ADC 또는 서비스 계정 JSON)

### 설치 & 실행

```bash
# (선택) 가상환경
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install --upgrade pip
pip install streamlit langchain langchain-openai langchain-core huggingface_hub google-genai
# Bedrock 사용 시:
# pip install langchain-aws

# 앱 실행
streamlit run app.py
