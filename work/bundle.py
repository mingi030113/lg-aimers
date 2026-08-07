"""submit.zip 생성기.

features.py 를 script_template.py 안에 인라인해 단일 script.py 를 만든 뒤
model/ + script.py + requirements.txt 만 담은 zip 을 만든다 (최상위 폴더 없음).
"""
import io
import os
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
# 학습에 쓴 버전과 동일하게 고정. 평가서버 기본 설치 패키지(numpy 1.26.4 등)와 충돌 없음.
REQUIREMENTS = "lightgbm==4.7.0\n"


def _strip(path):
    """모듈 docstring 과 numpy/pandas import 를 제거한 본문 (템플릿에 이미 있으므로)."""
    src = open(os.path.join(HERE, path), encoding="utf-8").read()
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)
    # 인라인되면 script.py 자체가 __main__ 이라 자체검증 블록이 실행돼 버린다. 잘라낸다.
    src = re.split(r"^if __name__ == ['\"]__main__['\"]:", src, maxsplit=1, flags=re.M)[0]
    return "\n".join(l for l in src.split("\n")
                     if not re.match(r"^(import numpy|import pandas)", l)).strip()


INLINE = ("features.py", "prior_stats.py", "qt_numpy.py")


def make_script():
    # 학습·추론이 완전히 같은 코드를 쓰도록 모듈들을 인라인한다
    body = "\n\n\n".join(_strip(f) for f in INLINE)
    tpl = open(os.path.join(HERE, "script_template.py"), encoding="utf-8").read()
    return tpl.replace("__FEATURES_SOURCE__", body)


def main(out="submit.zip"):
    script = make_script()
    with open(os.path.join(HERE, "script.py"), "w", encoding="utf-8") as f:
        f.write(script)

    model_dir = os.path.join(HERE, "model")
    files = sorted(os.listdir(model_dir))
    with zipfile.ZipFile(os.path.join(HERE, out), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("script.py", script)
        z.writestr("requirements.txt", REQUIREMENTS)
        for fn in files:
            z.write(os.path.join(model_dir, fn), f"model/{fn}")
    size = os.path.getsize(os.path.join(HERE, out))
    print(f"{out}  {size/1e6:.1f} MB")
    with zipfile.ZipFile(os.path.join(HERE, out)) as z:
        for i in z.infolist():
            print(f"  {i.filename:28s} {i.file_size/1e6:7.2f} MB")


if __name__ == "__main__":
    main()
