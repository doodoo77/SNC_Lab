from huggingface_hub import login
from datasets import load_dataset, Image, Sequence

# 1) 로그인 (토큰 하드코딩 대신 환경변수 HF_TOKEN도 가능)
login(token="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")

# 2) JSONL 로드 (폴더가 아니라 파일을 지정)
dsd = load_dataset("json", data_files="metadata.jsonl")
ds  = dsd["train"]

# 3) 경로 정규화: ./ 제거, 역슬래시를 슬래시로
def _clean_paths(ex):
    ex["file_names"] = [p.replace("\\", "/").lstrip("./") for p in ex["file_names"]]
    return ex
ds = ds.map(_clean_paths)

# 4) 컬럼 이름을 관례적으로 'images'로 변경(선택이지만 가독성↑)
ds = ds.rename_column("file_names", "images")

# 5) 멀티 이미지로 캐스팅 (핵심!)
ds = ds.cast_column("images", Sequence(Image()))

# (선택) 확인
print(ds.features)
# 기대: {'images': Sequence(feature=Image()), 'output': {...}}

# 6) 다시 허브로 푸시 (같은 리포로 덮어쓰기 커밋)
ds.push_to_hub("doodoo77/a11y-error-dataset-kor")
