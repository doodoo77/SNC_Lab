# convert_to_llamafactory.py
# 기능 요약:
# - JSONL/NDJSON/JSON 배열/}{로 붙은 케이스 모두 견고 파싱
# - LangChain PromptTemplate 사용 (langchain_core.prompts)
# - 빈 필드(검사항목/오류유형 등) 자동 보정(간단 휴리스틱)
# - 이미지 세트 기준 중복 방지(기본 활성화)
# - 변환 통계 출력

import json
import argparse
from pathlib import Path

# ---------------- LangChain PromptTemplate ----------------
try:
    from langchain_core.prompts import PromptTemplate
except ImportError as e:
    raise SystemExit(
        "LangChain이 없습니다. 다음으로 설치하세요:\n"
        "  python -m pip install -U langchain langchain-core langchain-community\n"
        "설치 후 다시 실행하세요."
    )

USER_PROMPT = PromptTemplate(
    template="""
    너는 접근성 평가 전문가야.

    진단할 페이지의 전체 및 오류로 의심되는 영역의 스크린샷과 오류 영역 코드를 주면,
    너는 접근성 진단 결과(검사항목/오류유형, 문제점 및 개선방안)를 도출하면 돼.

    해당 오류 영역이 위반한 검사항목/오류유형을 작성하고,
    문제점 및 개선방안은 해당 검사항목/오류유형을 준수하기 위해 사용자들에게 설명하는 설명문 혹은 코드를 작성해.

    또 위와 같은 진단 결과를 내기 전에 왜 그러한 진단이 나와야 하는지에 대해 추론해야 해.
    아래와 같은 절차를 따라 추론해.
    [추론 지침]
    1. 전체 페이지 스크린샷을 통해 해당 페이지의 목적을 파악하고,
    2. 페이지 목적을 참고할 때, 오류 영역 스크린샷에 드러난 진단 콘텐츠의 역할을 파악.
    3. 이제 오류 영역 코드까지 함께 고려할 때, 진단 결과 작성

    오류 영역 코드: {error_code}
    """,
    input_variables=["error_code"],
)

ASSISTANT_PROMPT = PromptTemplate(
    template="""
    <|begin_of_thought|>
    {rationale}
    <|end_of_thought|>

    <|begin_of_solution|>
    [검사항목]: {test_item}
    [오류유형]: {error_type}
    [문제점 및 개선방안_텍스트]: {text}
    [문제점 및 개선방안_코드]: {code}
    <|end_of_solution|>
    """,
    input_variables=["rationale", "test_item", "error_type", "text", "code"],
)

# ---------------- Robust loader ----------------
def iter_records(path: Path):
    """
    입력을 안전하게 순회:
      1) 정상 JSONL: 한 줄 = 한 객체
      2) 줄바꿈 없이 붙은 객체들: {...}{...}{...}
      3) 파일 전체가 JSON 배열: [ {...}, {...} ]
    """
    raw = path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    if not raw:
        return

    # 3) JSON 배열
    if raw.startswith("[") and raw.endswith("]"):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
                return
        except json.JSONDecodeError:
            pass  # 계속 진행

    # 1) 정상 JSONL 먼저 시도
    all_ok = True
    for idx, line in enumerate(raw.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            yield json.loads(s)
        except json.JSONDecodeError:
            all_ok = False
            break
    if all_ok:
        return

    # 2) }{로 붙은 객체들 분리(문자열/이스케이프 고려)
    buf, depth, in_str, esc = [], 0, False, False
    for ch in raw:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = "".join(buf).strip()
                    buf = []
                    start = chunk.find("{")
                    end = chunk.rfind("}")
                    if start != -1 and end != -1:
                        piece = chunk[start:end + 1]
                        try:
                            yield json.loads(piece)
                        except json.JSONDecodeError:
                            # 불가하면 스킵
                            pass

# ---------------- Field inference (heuristics) ----------------
def _norm(s: str) -> str:
    return (s or "").lower()

def infer_missing_fields(out: dict) -> dict:
    """
    검사항목/오류유형/문제점이 비었을 때 간단한 키워드로 보정.
    - 대비/contrast/#hex → 1.4.3 텍스트 명도대비(AA) / 텍스트-배경 대비 부족
    - 링크 텍스트/다운로드/MORE/목적 → 2.4.4 적절한 링크 텍스트 / 목적이나 용도를 알기 어려운 링크 텍스트
    """
    text_blob = " ".join([
        out.get("추론", "") or "",
        out.get("문제점", "") or "",
        out.get("문제점 및 개선방안_텍스트", "") or "",
        out.get("문제점 및 개선방안_코드", "") or "",
    ])
    n = _norm(text_blob)

    test_item = (out.get("검사항목") or "").strip()
    error_type = (out.get("오류유형") or "").strip()
    issue_text = (out.get("문제점") or "").strip()

    # 대비 관련
    contrast_hit = any(k in n for k in ["명도대비", "contrast", "#", "대비비", "ratio"])
    # 링크텍스트 관련
    link_hit = any(k in n for k in ["링크 텍스트", "link purpose", "more", "다운로드", "목적", "링크 목적"])

    if not test_item and contrast_hit:
        test_item = "1.4.3 텍스트 명도대비(AA)"
    if not error_type and contrast_hit:
        error_type = "텍스트-배경 대비 부족"

    if not test_item and link_hit:
        test_item = "2.4.4 적절한 링크 텍스트"
    if not error_type and link_hit:
        error_type = "목적이나 용도를 알기 어려운 링크 텍스트"

    # issue_text가 비었는데 대비 관련 수치가 보이면 간단 요약 생성
    if not issue_text and contrast_hit:
        # 간단히 색상/대비 키워드를 찾아 요약
        issue_text = "텍스트-배경 대비 부족 (명도대비 기준 미달)"

    out["검사항목"] = test_item
    out["오류유형"] = error_type
    out["문제점"] = issue_text
    return out

# ---------------- Builders ----------------
def build_user_content(error_code: str, num_images: int) -> str:
    image_tokens = "<image>" * max(num_images, 0)
    return (image_tokens + USER_PROMPT.format(error_code=error_code or "")).strip()

def build_assistant_content(out: dict) -> str:
    return ASSISTANT_PROMPT.format(
        rationale=out.get("추론", "") or "",
        test_item=out.get("검사항목", "") or "",
        error_type=out.get("오류유형", "") or "",
        text=out.get("문제점 및 개선방안_텍스트", "") or "",
        code=out.get("문제점 및 개선방안_코드", "") or "",
    ).strip()

def to_images_key(files):
    """중복 제거용 키 생성 (튜플)."""
    return tuple(files or [])

def convert_record(src: dict) -> dict:
    files = src.get("file_names", []) or src.get("images", []) or []
    out = src.get("output", {}) or {}

    # 필드 보정
    out = infer_missing_fields(out)

    user_content = build_user_content(out.get("문제점", "") or "", len(files))
    assistant_content = build_assistant_content(out)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": files,
    }

# ---------------- CLI ----------------
def main(inp: Path, outp: Path, dedup: bool = True):
    seen = set()
    total_in, total_out, skipped_dups, failed = 0, 0, 0, 0

    with outp.open("w", encoding="utf-8") as fout:
        for rec in iter_records(inp):
            total_in += 1
            try:
                # 중복 방지 (이미지 세트 기준)
                key = to_images_key(rec.get("file_names", []) or rec.get("images", []) or [])
                if dedup and key:
                    if key in seen:
                        skipped_dups += 1
                        continue
                    seen.add(key)

                converted = convert_record(rec)
                fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                total_out += 1
            except Exception as e:
                failed += 1
                # 경고만 출력하고 다음 레코드 진행
                print(f"[WARN] 레코드 {total_in} 변환 실패: {e}")

    # 통계 요약
    print("=== 변환 요약 ===")
    print(f"- 입력 레코드 수: {total_in}")
    print(f"- 출력 레코드 수: {total_out}")
    print(f"- 중복 스킵: {skipped_dups}")
    print(f"- 실패: {failed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert JSON(L/NDJSON/Array) to messages/images format (robust + dedup + field inference).")
    ap.add_argument("-i", "--input", type=Path, required=True, help="원본 JSONL/JSON 배열 파일 경로 (metadata.jsonl)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="출력 JSONL 경로 (messages.jsonl)")
    ap.add_argument("--no-dedup", action="store_true", help="이미지 세트 기준 중복 제거 비활성화")
    args = ap.parse_args()

    main(args.input, args.output, dedup=not args.no_dedup)
