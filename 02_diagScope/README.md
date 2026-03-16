# 🧭 AI 기반 웹 페이지 진단 대상 표집 도구

이 프로젝트는 **웹 접근성 진단 대상 화면을 자동으로 선정**하기 위한 파이프라인을 제공합니다.  
Selenium을 이용해 사이트 전체를 크롤링하고, Vision-LLM(OpenAI GPT-4o)을 활용하여 페이지 계층 트리와 라벨(테이블, 리스트, 이미지, 이미지 팝업 등)을 추출합니다.  
최종 결과는 **엑셀 파일(xlsx)** 로 저장되며, Streamlit 웹 UI를 통해 손쉽게 사용할 수 있습니다.

---

## 📌 Pipeline
![Pipeline](.\assets\diagScope.png)

---

## 📂 구성 파일

- **`crawler_ai.py`**  
  - Selenium으로 같은 도메인 내 모든 페이지 URL을 BFS 탐색  
  - 각 페이지에서 주요 HTML 요소 추출 (`breadcrumb`, `table`, `list`, `img`, `popup`)  
  - Vision-LLM을 통해 계층 경로(`path`)와 라벨(`label`) 자동 생성  
  - 결과를 `[계층 트리, URL, 라벨]` 형태로 DataFrame 구성 후 Excel 저장  

- **`streamlit_crawler.py`**  
  - Streamlit 기반 웹 인터페이스 제공  
  - 입력: 메인 URL  
  - 절차:  
    1. 크롤링으로 하위 메뉴/페이지 탐색  
    2. 페이지별 메뉴 계층 트리 추출 및 라벨링  
    3. 엑셀 파일(`sitemap_output.xlsx`) 다운로드 버튼 제공  
  - 실시간 진행 상황(크롤링 개수, 라벨링 진행률)을 화면에 표시  

- **`requirements.txt`**  
  - 필수 라이브러리:  
    - `selenium`, `openai`, `pandas`, `tqdm`, `streamlit`, `langchain`, `tiktoken` 등  

---

## ⚙️ 설치

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

**환경 변수 설정**  
```bash
export OPENAI_API_KEY=your_api_key_here
```

---

## 🚀 사용 방법

### 1) CLI 실행
```bash
python crawler_ai.py --start https://example.com --out sitemap_output.xlsx
```
- `--start`: 시작 URL  
- `--out`: 결과 엑셀 파일명 (기본값: `sitemap_output.xlsx`)  

### 2) Streamlit 실행
```bash
streamlit run streamlit_crawler.py
```
- 브라우저에서 UI 접속 (기본: `http://localhost:8501`)  
- 메인 URL 입력 후 버튼 클릭 → 탐색 & 라벨링 진행 → 엑셀 다운로드  

---

## 📊 출력 예시

| 계층 트리                | URL                      | 라벨     |
|---------------------------|--------------------------|----------|
| 메인 > 제품소개 > 상세    | https://.../product/123 | 테이블   |
| 메인 > 고객지원 > FAQ     | https://.../faq         | 리스트   |
| 메인 > 갤러리 > 이미지뷰  | https://.../gallery     | 이미지 팝업 |

---

## 🛡️ 주의사항
- Selenium 실행 시 크롬 드라이버 환경이 필요합니다 (`chromedriver-binary` 자동 설치 지원).  
- 페이지 수가 많을 경우 시간이 오래 걸릴 수 있습니다.  
- Vision-LLM 호출에는 OpenAI API 비용이 발생합니다.  

---

## 📎 참고
- **LLM 모델**: GPT-4o-mini (기본), 필요시 다른 모델로 변경 가능  
- **출력 형식**: Excel(`.xlsx`)  
- **주요 기능**: 사이트맵 자동 생성 + 라벨링 기반 진단 대상 선정
