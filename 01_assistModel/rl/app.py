# app.py
import os
import io
import json
import time
import uuid
import base64
import hashlib
import pathlib
from typing import Optional, List, Dict, Any
import pandas as pd
import re

import streamlit as st

# LangChain (OpenAI-호환/Bedrock용)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate  # LangChain 0.2+

# (옵션) AWS Bedrock을 LangChain으로 쓰려는 경우
try:
    from langchain_aws import ChatBedrock
    BEDROCK_AVAILABLE = True
except Exception:
    BEDROCK_AVAILABLE = False


# ========================= 공통 유틸 =========================
def b64_data_url(image_bytes: bytes, mime="image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

EXPORT_DIR = "hf_export"
EXPORT_JSONL = f"{EXPORT_DIR}/data.jsonl"
EXPORT_IMG_DIR = f"{EXPORT_DIR}/images"

def init_state():
    ss = st.session_state
    ss.setdefault("chat", [])                 # [{"role":"ai","raw":str,"data":dict}]
    ss.setdefault("history", [])
    ss.setdefault("last_ai_id", None)
    # 연결 상태 보존
    ss.setdefault("llm", None)
    ss.setdefault("provider_sel", None)
    ss.setdefault("model_name_sel", "")
    ss.setdefault("vertex_cfg", {})
    # 최근 JSON 결과(전문가 폼 채우기용)
    ss.setdefault("last_ai_json", None)
    ss.setdefault("last_ai_raw", "")

init_state()
ensure_dir(EXPORT_DIR)
ensure_dir(EXPORT_IMG_DIR)


# ========================= LLM 팩토리 =========================
@st.cache_resource(show_spinner=False)
def make_openai_like_llm(api_key: str, model: str, base_url: Optional[str], temperature: float):
    """OpenAI-호환 엔드포인트(예: OpenAI, Azure-OpenAI, 자체 호환 서버 등)."""
    if not api_key:
        raise ValueError("API Key가 필요합니다.")
    # JSON 모드 강제
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url or None,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

@st.cache_resource(show_spinner=False)
def make_bedrock_llm(region: str, model_id: str, temperature: float):
    """(옵션) AWS Bedrock. 사전 자격 증명 필요(AWS CLI/환경변수 등)."""
    if not BEDROCK_AVAILABLE:
        raise RuntimeError("langchain_aws 가 설치되어 있지 않습니다.")
    # Bedrock은 모델별 JSON 모드가 다르므로 여기서는 일반 설정만.
    return ChatBedrock(
        model_id=model_id,
        region_name=region,
        model_kwargs={"temperature": temperature},
    )

from google.genai import types

# ---------- Vertex AI(Gemini) 튜닝 엔드포인트 어댑터 ----------
def make_vertex_endpoint_llm(project_id: str, location: str, endpoint_id: str, credentials=None):
    from google import genai
    from google.genai.types import HttpOptions

    class VertexEndpointLLM:
        def __init__(self, project: str, loc: str, eid: str, creds):
            if not (project and loc and eid):
                raise ValueError("PROJECT_ID / LOCATION / ENDPOINT_ID가 필요합니다.")
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=loc,
                credentials=creds,
                http_options=HttpOptions(api_version="v1"),
            )
            self.model = f"projects/{project}/locations/{loc}/endpoints/{eid}"

        def generate(self, user_text: str, *, image_bytes: bytes | None = None,
                     mime: str | None = None, system_prompt: str | None = None) -> str:
            parts = []
            if system_prompt:
                parts.append({"text": f"[SYSTEM]\n{system_prompt}"})
            parts.append({"text": user_text})
            if image_bytes:
                parts.append({"inline_data": {"mime_type": mime or "image/png", "data": image_bytes}})
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": parts}],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            return getattr(resp, "text", str(resp))

        def invoke(self, messages):
            from langchain_core.messages import SystemMessage, HumanMessage
            system_prompt, user_text = "", ""
            image_bytes, mime = None, None
            for m in messages:
                if isinstance(m, SystemMessage) and isinstance(m.content, str):
                    system_prompt += m.content
                elif isinstance(m, HumanMessage):
                    if isinstance(m.content, str):
                        user_text += m.content
                    elif isinstance(m.content, list):
                        for c in m.content:
                            if c.get("type") == "text":
                                user_text += c.get("text", "")
                            elif c.get("type") == "image_url":
                                url = c["image_url"]["url"]
                                if url.startswith("data:"):
                                    header, b64 = url.split(",", 1)
                                    mime = header.split(";")[0].split(":")[1]
                                    image_bytes = base64.b64decode(b64)
            return self.generate(user_text, image_bytes=image_bytes, mime=mime,
                                 system_prompt=system_prompt)

    return VertexEndpointLLM(project_id, location, endpoint_id, credentials)

# ========================= 공통 모델 호출(멀티모달) =========================
def call_llm_with_optional_image(llm, user_text: str, image_bytes: Optional[bytes]) -> str:
    """
    멀티모달 메시지 전송. (JSON 모드로 응답하도록 프롬프트에서 강제)
    """
    if image_bytes:
        data_url = b64_data_url(image_bytes)
        human = HumanMessage(content=[
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])
    else:
        human = HumanMessage(content=user_text)

    sys = SystemMessage(content="You are a helpful assistant. Reply with pure JSON only.")
    ai = llm.invoke([sys, human])
    return ai.content if isinstance(ai, AIMessage) else str(ai)


# ========================= JSON 파싱/렌더링 =========================
def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """모델이 마크다운/설명을 섞어도 JSON 본문만 추출해서 파싱."""
    # 코드펜스/불순물 제거
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # 가장 바깥 {} 블럭만 잡기
        m = re.search(r"\{[\s\S]*\}\s*$", cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

def render_result(data: Dict[str, Any], raw_text: str | None = None):
    checks = data.get("checks", {})
    fix = data.get("fix", {})
    reasoning = data.get("reasoning", [])

    with st.container(border=True):
        # 1) 🔎 추론(요약) — 제일 위에, 접지 않고 바로 표시
        if reasoning:
            st.markdown("### 🔎 추론(요약)")
            for i, r in enumerate(reasoning, 1):
                st.markdown(f"{i}. {r}")
            st.markdown("---")  # 추론과 진단 결과 사이 구분선

        # 2) ✅ 진단 결과
        st.markdown("### ✅ 진단 결과")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**검사항목**")
            st.write(checks.get("검사항목", ""))
        with col2:
            st.markdown("**오류유형**")
            st.write(checks.get("오류유형", ""))

        st.markdown("**개선방안(설명)**")
        st.write(fix.get("text", ""))

        if fix.get("code_html"):
            st.markdown("**개선방안(코드)**")
            st.code(fix["code_html"], language="html")

        # 3) 📄 모델 응답 전체(JSON) — 이건 그대로 접어 두기
        if raw_text:
            with st.expander("📄 모델 응답 전체(JSON)", expanded=False):
                st.code(raw_text, language="json")




# ========================= JSONL 레코드 =========================
def build_record(
    *,
    model_text_original: str,
    model_text_edited: str,
    feedback_score: Optional[int],
    feedback_comment: Optional[str],
    model_name: str,
    image_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    rec_id = str(uuid.uuid4())
    return {
        "id": rec_id,
        "ts": int(time.time()),
        "model_name": model_name,
        "model_text_original": model_text_original,  # JSON 문자열
        "model_text_edited": model_text_edited,      # JSON 문자열(편집본)
        "feedback_score": feedback_score,            # 1~5
        "feedback_comment": feedback_comment or "",
        "image": image_meta or {},                   # {"path": "...", "sha256": "...", "mime": "..."}
    }

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    ensure_dir(str(pathlib.Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ========================= 사이드바(모델 설정) =========================
st.sidebar.header("🔧 모델 설정")
provider = st.sidebar.selectbox(
    "Provider",
    ["OpenAI-compatible", "AWS Bedrock", "Vertex AI (Gemini Endpoint)"]
)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.2, 0.1)

llm = None
model_name = ""

if provider == "OpenAI-compatible":
    api_key = st.sidebar.text_input("API Key", type="password")
    base_url = st.sidebar.text_input("Base URL (선택)", help="OpenAI-호환 서버의 엔드포인트가 있으면 입력")
    model_name = st.sidebar.text_input("Model", value="gpt-4o-mini")
    if st.sidebar.button("🔌 Connect", use_container_width=True):
        try:
            llm = make_openai_like_llm(api_key, model_name, base_url, temperature)
            st.sidebar.success("연결 성공")
        except Exception as e:
            st.sidebar.error(f"연결 실패: {e}")
    else:
        if api_key and model_name:
            try:
                llm = make_openai_like_llm(api_key, model_name, base_url, temperature)
            except Exception:
                pass

elif provider == "AWS Bedrock":
    region = st.sidebar.text_input("AWS Region", value="us-east-1")
    model_name = st.sidebar.text_input("Model ID", value="anthropic.claude-3-5-sonnet-20241022-v2:0")
    if st.sidebar.button("🔌 Connect", use_container_width=True):
        try:
            llm = make_bedrock_llm(region, model_name, temperature)
            st.sidebar.success("연결 성공")
        except Exception as e:
            st.sidebar.error(f"연결 실패: {e}")

elif provider == "Vertex AI (Gemini Endpoint)":
    from google.oauth2 import service_account
    import json as pyjson

    st.sidebar.markdown("튜닝된 **Gemini 2.5 Pro** 엔드포인트를 지정하세요.")
    project_id  = st.sidebar.text_input("PROJECT_ID",  value="",        key="vx_project_id")
    location    = st.sidebar.text_input("LOCATION",    value="us-central1", key="vx_location")
    endpoint_id = st.sidebar.text_input("ENDPOINT_ID", value="",        key="vx_endpoint_id")

    # ▶ 서비스 계정 JSON 업로드(선택)
    with st.sidebar.expander("Google 인증(서비스 계정 JSON 업로드)", expanded=False):
        sa_file = st.file_uploader("service-account.json", type=["json"], key="vx_sa_json")
        creds = None
        if sa_file is not None:
            sa_info = pyjson.loads(sa_file.getvalue())
            creds = service_account.Credentials.from_service_account_info(
                sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            st.sidebar.success("서비스 계정 로드 완료")

    model_name = endpoint_id or "vertex-gemini-endpoint"

    if st.sidebar.button("🔌 Connect", use_container_width=True, key="vx_connect_btn"):
        try:
            llm = make_vertex_endpoint_llm(project_id, location, endpoint_id, credentials=creds)
            st.sidebar.success("Vertex 엔드포인트 연결 성공")
            # 세션에 보존(재실행 대비)
            st.session_state.llm = llm
            st.session_state.provider_sel = "Vertex AI (Gemini Endpoint)"
            st.session_state.model_name_sel = model_name
            st.session_state.vertex_cfg = {
                "project": project_id, "location": location, "endpoint": endpoint_id
            }
        except Exception as e:
            st.sidebar.error(f"연결 실패: {e}")


# ========================= 본문 UI =========================
st.title("접근성 진단 능력 평가 페이지")
st.caption("이미지 입력, 외부 엔드포인트 호출, JSON 구조화, 전문가 피드백, JSONL 저장을 포함합니다.")

# 1) 유저 입력
with st.container(border=True):
    st.subheader("입력")

    # 1) 오류 영역 이미지 업로드
    with st.expander("📎 오류 영역 이미지 업로드", expanded=False):
        uploaded_img = st.file_uploader("오류 영역 이미지 업로드 (선택)", type=["png", "jpg", "jpeg", "webp"])

    # 2) 표준 개선방안 리스트(Excel) 업로드
    standard_texts_str = ""
    std_rows_count = 0
    with st.expander("📎 표준 개선방안 리스트(Excel) 업로드", expanded=False):
        std_file = st.file_uploader("표준 개선방안 Excel (.xlsx/.xls)", type=["xlsx", "xls"], key="std_xlsx")
        if std_file is not None:
            try:
                xls = pd.ExcelFile(std_file)
                sheet_sel = st.selectbox("시트 선택", xls.sheet_names, key="std_sheet_sel")
                df = xls.parse(sheet_sel)

                st.caption("미리보기 (상위 10행)")
                st.dataframe(df.head(10), use_container_width=True)

                cols = st.multiselect("포함할 열 선택", list(df.columns), default=list(df.columns), key="std_cols")
                max_rows = st.slider("포함할 최대 행 수", 10, 2000, 300, step=10, key="std_max_rows")

                df_use = df[cols] if cols else df
                df_use = df_use.head(max_rows)
                std_rows_count = len(df_use)

                records = df_use.to_dict(orient="records")
                standard_texts_str = json.dumps(records, ensure_ascii=False)

                st.caption(f"모델에 전달될 표준 텍스트 (총 {std_rows_count}행)")
                st.text_area(
                    "전달 본문(읽기 전용, 토큰 절약을 위해 열/행을 조절하세요)",
                    value=standard_texts_str[:10000],
                    height=160,
                    disabled=True,
                    key="std_preview",
                )
            except Exception as e:
                st.warning(f"엑셀 처리 중 오류: {e}")
        else:
            standard_texts_str = ""

    # 3) 오류 영역 코드 입력
    error_code_str = st.text_area("오류 영역 코드", value="", height=220, key="err_code_text")

    # 4) 전문가 메모
    memo_str = st.text_area("전문가 메모", placeholder="진단에 도움되는 맥락/특이사항 등을 메모하세요.", height=120, key="expert_memo")

    # 5) 버튼
    c1, c2 = st.columns([1,1])
    with c1:
        run_btn = st.button("모델 호출", use_container_width=True)
    with c2:
        clear_btn = st.button("대화 초기화", use_container_width=True)

if clear_btn:
    st.session_state.chat.clear()
    st.session_state.last_ai_id = None
    st.session_state.last_ai_json = None
    st.rerun()

llm = st.session_state.llm
active_provider = st.session_state.provider_sel or provider
model_name = st.session_state.model_name_sel or model_name  # 기록용

# 2) 모델 호출
if run_btn:
    if llm is None:
        st.error("사이드바에서 모델 연결 정보를 입력/연결하세요.")
    else:
        # --- 접근성 평가 자동 프롬프트 (JSON 강제) ---
        A11Y_PROMPT = r"""[[역할]
        너는 접근성 평가 전문가다.

        [입력]
        전체 페이지 스크린샷 -
        오류 영역 스크린샷 -  
        표준 개선방안 리스트 - {standard_texts}
        오류 영역 코드 - {error_code}
        인간 전문가가 작성한 메모 - {memo}

        [지시문]
        1) 먼저 [오류 영역 스크린샷/설명]이 위반한 접근성 오류 유형을 판단하고,
        [표준 개선방안 리스트]에서 **가장 관련 있는 항목의 “검사항목”과 “오류유형” 텍스트를 그대로 인용**해.

        2) “문제점 및 개선방안”을 작성해.
        - _텍스트_: 왜 문제인지 + 표준 충족을 위해 어떻게 코드가 수정되어야 하는지 쉬운 문장으로 장황하지 않으면서 핵심적인 내용을 포함해서 설명.
        - _코드(선택)_: {error_code}가 주어졌다면, 해당 오류를 준수하기 위해서 수정되어야 할 코드를 제시하면 돼.

        3) 진단 전 필수 추론 절차:
        - [전체 페이지 스크린샷/설명]으로 페이지 목적 1줄 파악
        - 그 목적을 참고하여 [오류 영역 스크린샷/설명]의 콘텐츠 역할 파악
        - {error_code}까지 고려해 최종 진단 작성
        - 아래 명시된 자제검증 체크리스트에 부합될때까지 추론 과정을 반복해

        [출력 형식 — JSON만, 한국어, 마크다운/설명/코드펜스 금지]
        {% raw %}
        {
        "reasoning": ["핵심 추론 1", "핵심 추론 2"],
        "checks": { "검사항목": "<표준 인용>", "오류유형": "<표준 인용>" },
        "fix": { "text": "개선방안 설명", "code_html": "<수정 예시 HTML 또는 빈 문자열>" }
        }
        {% endraw %}
        """
        prompt_tmpl = PromptTemplate(template=A11Y_PROMPT, template_format="jinja2")
        combined_text = prompt_tmpl.format(
            standard_texts=standard_texts_str or "",
            error_code=error_code_str or "",
            memo=memo_str or "",
        )

        image_bytes = uploaded_img.read() if uploaded_img else None
        try:
            if active_provider == "Vertex AI (Gemini Endpoint)":
                mime = (uploaded_img.type if uploaded_img and hasattr(uploaded_img, "type") else None)
                ai_text = llm.generate(
                    combined_text,
                    image_bytes=image_bytes,
                    mime=mime,
                    system_prompt="Reply with pure JSON only."
                )
            else:
                ai_text = call_llm_with_optional_image(llm, combined_text, image_bytes)
        except Exception as e:
            st.error(f"모델 호출 실패: {e}")
            ai_text = ""

        st.session_state.last_ai_raw = ai_text

        # JSON 파싱
        data = safe_json_loads(ai_text) if ai_text else None
        if not data:
            st.warning("모델이 JSON 형식을 따르지 않았습니다. 아래 원문 응답을 참고하세요.")
            st.session_state.last_ai_json = None
        else:
            # 결과는 세션에만 저장(렌더링은 아래 공통 섹션에서 한 번만)
            st.session_state.last_ai_json = data

        # 이미지 저장(있다면) - 메타 기록용
        image_meta = None
        if image_bytes:
            img_id = str(uuid.uuid4())
            ext = pathlib.Path(uploaded_img.name).suffix.lower() or ".png"
            img_path = f"{EXPORT_IMG_DIR}/{img_id}{ext}"
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            image_meta = {"path": img_path}
            try:
                image_meta["sha256"] = sha256_bytes(image_bytes)
                image_meta["mime"] = uploaded_img.type if hasattr(uploaded_img, "type") else "image/*"
            except Exception:
                pass

        # 채팅 타임라인에는 AI만 보존
        st.session_state.chat.append({"role": "ai", "raw": ai_text, "data": data or None})
        st.session_state.last_ai_id = len(st.session_state.chat) - 1


# 3) 진단 결과 + 추론/원문 출력(최근 1개만)
if st.session_state.last_ai_json or st.session_state.last_ai_raw:
    with st.chat_message("assistant"):
        if st.session_state.last_ai_json:
            # JSON 파싱 성공: 카드 + 추론 + JSON expander
            render_result(
                st.session_state.last_ai_json,
                raw_text=st.session_state.last_ai_raw,
            )
        else:
            # JSON 파싱 실패: 원문만 표시
            st.write(st.session_state.last_ai_raw)


# 4) 전문가 검증/편집/피드백 (JSON을 폼에 자동 주입)
if st.session_state.last_ai_id is not None:
    ai_idx = st.session_state.last_ai_id
    ai_raw = st.session_state.chat[ai_idx].get("raw", "")
    ai_data = st.session_state.chat[ai_idx].get("data", None) or st.session_state.last_ai_json

    with st.container(border=True):
        st.subheader("전문가 피드백")

        # 1. 추론 먼저 편집
        st.markdown("#### 1. 추론 수정")
        f_reasoning = st.text_area(
            "추론(한 줄당 하나, 최대 5개 권장)",
            value="\n".join((ai_data or {}).get("reasoning", [])),
            height=140,
        )

        # 2. 진단 항목 (검사항목 / 오류유형)
        st.markdown("#### 2. 진단 항목 수정")
        checks_col1, checks_col2 = st.columns(2)
        with checks_col1:
            f_check_item = st.text_input(
                "검사항목",
                value=(ai_data or {}).get("checks", {}).get("검사항목", ""),
            )
        with checks_col2:
            f_check_type = st.text_input(
                "오류유형",
                value=(ai_data or {}).get("checks", {}).get("오류유형", ""),
            )

        # 3. 개선방안 (설명 / 코드)
        st.markdown("#### 3. 개선방안 수정")
        f_fix_text = st.text_area(
            "개선방안(설명)",
            value=(ai_data or {}).get("fix", {}).get("text", ""),
            height=140,
        )
        f_fix_code = st.text_area(
            "개선방안(코드, HTML만)",
            value=(ai_data or {}).get("fix", {}).get("code_html", ""),
            height=160,
        )

        # 4. 피드백 점수 · 코멘트 · 저장 버튼
        st.markdown("#### 4. 피드백")
        cA, cB, cC = st.columns([1, 2, 1])
        with cA:
            score = st.radio("만족도 점수", [1, 2, 3, 4, 5], index=3, horizontal=True)
        with cB:
            comment = st.text_input("코멘트(선택)", placeholder="왜 만족/불만족인지, 수정 이유 등")
        with cC:
            save_btn = st.button("📝 피드백 저장\n(JSONL에 추가)", use_container_width=True)

        if save_btn:
            # 편집본 JSON 조립
            edited_json = {
                "reasoning": [s.strip() for s in f_reasoning.split("\n") if s.strip()],
                "checks": {"검사항목": f_check_item, "오류유형": f_check_type},
                "fix": {"text": f_fix_text, "code_html": f_fix_code},
            }
            edited_str = json.dumps(edited_json, ensure_ascii=False)

            image_meta = None  # (이미지 메타는 필요시 여기에 연결)

            rec = build_record(
                model_text_original=ai_raw,
                model_text_edited=edited_str if edited_str != ai_raw else "",
                feedback_score=int(score),
                feedback_comment=comment or "",
                model_name=model_name,
                image_meta=image_meta,
            )
            st.session_state.history.append(rec)
            append_jsonl(EXPORT_JSONL, rec)
            st.success(f"저장 완료: {EXPORT_JSONL}")

            with st.expander("저장된 레코드 미리보기"):
                st.json(rec)


# 5) 내보내기 / 허깅페이스 업로드
with st.container(border=True):
    st.subheader("📦 데이터셋 내보내기")
    st.caption("`hf_export/data.jsonl` 파일에 자동 누적 저장됩니다. 허깅페이스 데이터셋으로 바로 push할 수 있습니다.")
    c1, c2 = st.columns([1,1])
    with c1:
        if pathlib.Path(EXPORT_JSONL).exists():
            with open(EXPORT_JSONL, "rb") as f:
                st.download_button("data.jsonl 다운로드", f, file_name="data.jsonl", mime="application/jsonl")
        else:
            st.info("아직 저장된 레코드가 없습니다.")
    with c2:
        st.write(f"저장 경로: `{EXPORT_JSONL}`")

    with st.expander("허깅페이스 허브로 업로드(선택)"):
        from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
        repo_id = st.text_input("repo_id (org/name)", placeholder="your-org/your-dataset")
        hf_token = st.text_input("HF_TOKEN", type="password")
        path_in_repo = st.text_input("경로(리포 내)", value="data/data.jsonl")
        private = st.checkbox("비공개로 생성", value=True)
        do_push = st.button("⬆️ Push to Hub")

        if do_push:
            if not (repo_id and hf_token and pathlib.Path(EXPORT_JSONL).exists()):
                st.error("repo_id, HF_TOKEN, data.jsonl 경로를 확인하세요.")
            else:
                try:
                    HfFolder.save_token(hf_token)
                    try:
                        create_repo(repo_id, token=hf_token, private=private, repo_type="dataset")
                    except Exception:
                        pass  # 이미 존재
                    upload_file(
                        path_or_fileobj=EXPORT_JSONL,
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token,
                    )
                    st.success(f"Hugging Face Hub 업로드 완료: {repo_id} / {path_in_repo}")
                except Exception as e:
                    st.error(f"업로드 실패: {e}")
