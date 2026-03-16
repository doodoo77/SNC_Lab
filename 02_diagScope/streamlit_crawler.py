# streamlit_app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from io import BytesIO
from pathlib import Path

# ——— 기존 파이프라인 함수 임포트 ———
from crawler_ai import crawl_urls, extract_html, vision_extract
from langchain_openai import ChatOpenAI

st.markdown(
    """
    <style>
    /* subheader(h2) 스타일 통일 */
    h2 {
      font-size: 1.25rem !important;
      font-weight: 600 !important;
      font-family: 'Apple SD Gothic Neo', sans-serif !important;
    }
    /* spinner 텍스트 스타일 통일 (이제 직접 안 쓰지만 남겨둬도 OK) */
    div[role="status"] > span {
      font-size: 1.25rem !important;
      font-weight: 600 !important;
      font-family: 'Apple SD Gothic Neo', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="SNC Lab", layout="wide")
st.title("진단 대상 표집을 위한 AI")
st.markdown(
    """
    <div style="background-color:#f8f9fa;
                border:1px solid #e6e6e6;
                padding:16px 20px;
                border-radius:8px;
                line-height:1.6em;
                margin-bottom:24px;">
        해당 웹은 진단 대상 표집을 위해 아래와 같은 절차를 수행합니다.<br>
        1&#41; 주어진 메인 URL에서 <strong>모든 하위 메뉴를 탐색(크롤링)</strong>한 뒤,<br>
        2&#41; 페이지 별 <strong>메뉴 계층 트리 추출</strong> 및 <strong>라벨링</strong>(이미지, 이미지 팝업, 리스트, 테이블)하여,<br>
        3&#41; <strong>진단 대상 표집 범위</strong>라는 이름의 엑셀 파일을 제공합니다.
    </div>
    """,
    unsafe_allow_html=True
)

# 1) 입력 폼 --------------------------------------------------------------
start_url = st.text_input("🔗 메인 URL을 입력하세요", placeholder="https://example.com")

# 2) 실행 버튼 ------------------------------------------------------------
if st.button("🚀 표집 시작"):
    if not start_url.startswith("http"):
        st.error("유효한 URL을 입력해주세요.")
        st.stop()

    # ——— 플레이스홀더 생성 ———
    header_ph   = st.empty()
    progress_ph = st.empty()

    # ── 1단계. 표집 대상 탐색 ─────────────────────────
    header_ph.subheader("1단계. 표집 대상 탐색")
    progress_ph.text("🔍 탐색 중: 0개 페이지 발견")  # 초기 메시지

    urls = []
    for i, url in enumerate(crawl_urls(start_url), start=1):
        urls.append(url)
        # max_pages가 없으니, 발견 개수만 갱신
        progress_ph.text(f"🔍 탐색 중: {i}개 페이지 발견")

    st.success(f"✅ 크롤링 완료: 총 {len(urls)}개 페이지 발견")

    # ── 2단계. 메뉴 계층 트리 추출 및 라벨링 ───────────
    # → 같은 header_ph, progress_ph 에 덮어쓰기
    header_ph.subheader("2단계. 메뉴 계층 트리 추출 및 라벨링")
    label_prog = progress_ph.progress(0.0, text="0/… 라벨링 중")

    # LLM 한 번만 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=100)
    rows = []
    total = len(urls)

    for idx, url in enumerate(urls, start=1):
        html_dict = extract_html(url)
        html = "\n".join(sum(html_dict.values(), []))
        path, label = vision_extract(html, llm)

        rows.append({
            "인덱스": idx,
            "계층 트리": path,
            "URL": url,
            "라벨": label
        })

        label_prog.progress(idx / total, text=f"{idx}/{total} 라벨링 완료")

    st.success("✅ 라벨링 완료")

    # ── 결과 표시 & 다운로드 ─────────────────────────────
    df = pd.DataFrame(rows)
    # … 이하 생략

    # ── 결과 표시 & 다운로드 ─────────────────────────────────────────────
    df = pd.DataFrame(rows)
    st.subheader("🔍 결과 미리보기")
    st.dataframe(df, use_container_width=True)

    towrite = BytesIO()
    df.to_excel(towrite, index=False, sheet_name="sitemap")
    towrite.seek(0)

    st.download_button(
        label="📥 **진단 대상 표집 범위** 엑셀 파일 다운로드",
        data=towrite,
        file_name="sitemap_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.success("✅ 진단 완료! 엑셀 파일을 다운로드하세요.")
