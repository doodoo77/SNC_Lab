# -*- coding: utf-8 -*-
"""crawler_ai_pipeline.py
=========================
**목표**: 
1. Selenium으로 같은 도메인 내 모든 페이지 URL을 수집한다.
2. 각 URL을 차례로 열어 전체 페이지 스크린샷을 찍고, Vision‑LLM(OpenAI GPT‑4o)에게
   *계층 경로(path)* 과 *현재 메뉴 라벨(label)*, *페이지 유형(types)* 을 JSON 으로 추출하도록 요청한다.
3. 최종 결과를 **계층 트리 / URL / 라벨**(types 포함) 순서로 정리하여 Excel(xlsx)로 저장한다.

> ⚠️  필요 패키지 (conda/pip)
> ```bash
> pip install selenium openai pillow pandas tqdm webdriver-manager
> ```
> OPENAI API 키를 환경변수 `OPENAI_API_KEY` 로 지정하세요.

---
"""

import base64
import io
import os
import re
import time
import urllib.parse as urlparse
from collections import deque
from pathlib import Path

import openai
import pandas as pd
from PIL import Image

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

####################
#  Config params   #
####################
MAX_DEPTH = 6
MAX_PAGES = 50
PAGE_TIMEOUT = 15

import os
import streamlit as st

import tempfile
from collections import deque
from urllib.parse import urlparse

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Iterator

os.environ['OPENAI_API_KEY'] = st.secrets.get("OPENAI_API_KEY")
###############################################################################
#  Step 1: Crawl URLs with Selenium                                           #
###############################################################################

# 크롤링/추출에 사용할 기본 설정
PAGE_TIMEOUT = 30
MAX_PAGES = 300
MAX_DEPTH = 3

def crawl_urls(start_url: str) -> Iterator[str]:
    """
    start_url로부터 같은 도메인 내 링크를 BFS 방식으로
    depth 제한 없이, 페이지 제한 없이 모두 순회하며
    방문한 URL을 하나씩 yield.
    """
    from collections import deque
    from urllib.parse import urlparse
    import tempfile
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    parsed = urlparse(start_url)
    base   = f"{parsed.scheme}://{parsed.netloc}"

    tmp_profile_dir = tempfile.mkdtemp()
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=ko-KR")
    options.add_argument(f"--user-data-dir={tmp_profile_dir}")
    options.page_load_strategy = "eager"

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGE_TIMEOUT)
    driver.execute_cdp_cmd("Network.enable", {})
    driver.execute_cdp_cmd("Network.setBlockedURLs", {
        "urls": ["*.css","*.js","*.png","*.jpg","*.jpeg","*.svg","*.gif","*.woff2"]
    })

    visited = set()
    queue   = deque([start_url])

    try:
        while queue:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                driver.get(url)
            except Exception:
                continue

            yield url

            # 같은 도메인 내부 링크 수집
            for a in driver.find_elements(By.TAG_NAME, "a")[:100]:
                href = a.get_attribute("href") or ""
                if href.startswith(base) and href not in visited:
                    queue.append(href)
    finally:
        driver.quit()



from selenium.common.exceptions import TimeoutException

def extract_html(page_url: str) -> dict[str, list[str]]:
    """
    단일 페이지를 로드하고,
    breadcrumb, table, list, img, popup 요소들의 outerHTML을 딕셔너리로 반환.
    타임아웃 시에는 body 전체 outerHTML을 'body' 키로 반환.
    """
    # 임시 프로필 디렉토리
    tmp_profile_dir = tempfile.mkdtemp()

    # Chrome 옵션
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--lang=ko-KR")
    options.add_argument(f"--user-data-dir={tmp_profile_dir}")
    options.page_load_strategy = "eager"

    

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGE_TIMEOUT)

    # 리소스 차단
    driver.execute_cdp_cmd("Network.enable", {})
    driver.execute_cdp_cmd("Network.setBlockedURLs", {
        "urls": ["*.css", "*.js", "*.png", "*.jpg", "*.jpeg", "*.svg", "*.gif", "*.woff2"]
    })

    try:
        driver.get(page_url)

        try:
            # 주요 요소가 뜰 때까지 최대 10초 대기
            WebDriverWait(driver, 10).until(EC.presence_of_element_located(
                (By.CSS_SELECTOR,
                 ".breadcrumb, .nav-path, nav[aria-label='breadcrumb'], table, img")
            ))
            # 정상 로드된 경우 주요 셀렉터만 추출
            html = driver.execute_script("""
              const sel = {
                breadcrumb: ".breadcrumb, .nav-path, nav[aria-label='breadcrumb']",
                tables:     "table",
                lists:      "ul, ol, dl",
                images:     "img",
                popups:     ".modal, .lightbox, [data-lightbox], [aria-modal='true']"
              };
              const result = {};
              for (let key in sel) {
                const nodes = document.querySelectorAll(sel[key]);
                if (nodes.length) {
                  result[key] = Array.from(nodes).map(n => n.outerHTML);
                }
              }
              return result;
            """)
        except TimeoutException:
            # 타임아웃 시 경고 후 body 전체 HTML을 한 덩어리로 반환
            print(f"[WARN] Timeout waiting for selectors on {page_url}")
            fallback = driver.execute_script("return document.body.outerHTML;")
            html = {"body": [fallback]}

    finally:
        driver.quit()

    return html



###############################################################################
#  Step 2: Vision‑LLM label & type extractor                                  #
###############################################################################

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# -*- coding: utf-8 -*-
"""crawler_ai_pipeline.py ► 토큰 제한 로직 추가 버전"""

import tiktoken  # pip install tiktoken
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 모델별 최대 입력 토큰 수 (예시는 gpt-4o-mini용)
MODEL_MAX_INPUT_TOKENS = {
    "gpt-4o-mini": 8192,
    "gpt-4.1-nano-2025-04-14": 32768,
    # 필요시 다른 모델도 추가
}

def trim_to_limit(text: str, model_name: str, reserve_output: int = 512) -> str:
    """
    모델 최대 입력 토큰 수 - reserve_output 만큼만 입력 토큰으로 허용.
    넘으면 토큰 단위로 잘라서 반환.
    """
    max_tokens = MODEL_MAX_INPUT_TOKENS.get(model_name, 4000) - reserve_output
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    # 잘라낸 토큰을 다시 디코딩
    trimmed = enc.decode(tokens[:max_tokens])
    print(f"[WARN] input trimmed from {len(tokens)} to {max_tokens} tokens")
    return trimmed

def vision_extract(html: str, llm: ChatOpenAI):
    # 1) HTML 길이가 너무 길면 trim
    model_name = llm.model_name
    html = trim_to_limit(html, model_name, reserve_output=256)

    # 2) 파싱용 Pydantic 모델 정의
    class match_label(BaseModel):
        path: str = Field(description="현재 페이지의 계층 트리 구조를 파악해. 이를 위해서, 해당 페이지까지의 메뉴·탐색 경로를 `' > '` 로 연결한 문자열로 반환해.")
        label: str = Field(description="""해당 페이지가 아래에 정의된 label에 해당되는 모든 라벨값을 문자열로 반환해.
                            - "테이블"   : 표를 포함하는 페이지
                            - "리스트"   : ul/ol/dl 구조가 2 종류 이상인 페이지
                            - "이미지"   : 이미지를 포함하는 페이지
                            - "이미지 팝업" : 썸네일·버튼 클릭 시 모달·라이트박스로 확대 이미지를 가진 페이지
                            """
                        )

    label_parser = JsonOutputParser(pydantic_object=match_label)
    extractor_llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")  # 필요시 변경
    extractor = extractor_llm.with_structured_output(match_label)
    extractor_prompt = PromptTemplate(
        template="""
            주어진 웹 페이지의 html에 접속한 다음, 두가지 일을 수행해야 해.
            1. 현재 페이지의 계층 트리 구조를 파악해. 이를 위해서, 해당 페이지까지의 메뉴·탐색 경로를 `' > '` 로 연결한 문자열로 반환해.
            2. 현재 페이지의 라벨을 파악해. 아래에 정의된 label에 해당되는 모든 라벨값을 문자열로 반환해.
            - "테이블"   : 표를 포함하는 페이지
            - "리스트"   : ul/ol/dl 구조가 2 종류 이상인 페이지
            - "이미지"   : 이미지를 포함하는 페이지
            - "이미지 팝업" : 썸네일·버튼 클릭 시 모달·라이트박스로 확대 이미지를 가진 페이지

            현재 페이지 html: {html}

            답변을 생성할 때는 아래 지침을 따라.: {format_instructions}
        """,
        input_variables=["html"],
        partial_variables={"format_instructions": label_parser.get_format_instructions()}
    )

    # 3) trimmed HTML 로 호출
    result = extractor.invoke(extractor_prompt.format(html=html))
    return result.path, result.label

def parse_json(text: str):
    import json
    try:
        return json.loads(text)
    except Exception:
        return {"path": "(알 수 없음)", "label": "(알 수 없음)", "types": []}

###############################################################################
#  Step 3: Full pipeline                                                      #
###############################################################################

def build_sitemap(start_url: str, out_xlsx: Path):
    # generator 를 list 로 감싸서 전체 URL 리스트로
    urls = list(crawl_urls(start_url))

    # 이후에는 예전처럼 len(urls) 쓰면서 tqdm(total=len(urls)) 가능
    from tqdm import tqdm
    rows = []
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=100)
    with tqdm(total=len(urls), desc="Vision labeling") as bar:
        for url in urls:
            path, label = vision_extract("\n".join(sum(extract_html(url).values(), [])), llm)
            rows.append([path, url, label])
            bar.update(1)

    # Save to Excel
    df = pd.DataFrame(rows, columns=["계층 트리", "URL", "라벨" ])
    df.to_excel(out_xlsx, index=False)
    print("Saved", len(rows), "rows →", out_xlsx)

###############################################################################
#  CLI                                                                       #
###############################################################################
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="시작 URL")
    ap.add_argument("--out", default="sitemap_output.xlsx", help="결과 xlsx 파일명")
    args = ap.parse_args()

    build_sitemap(args.start, Path(args.out))

