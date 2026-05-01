from pathlib import Path
import re
import zlib


def decode_pdf_string(value: str) -> str:
    value = value.replace(r"\\", "\\")
    value = value.replace(r"\(", "(")
    value = value.replace(r"\)", ")")
    value = value.replace(r"\n", " ")
    value = value.replace(r"\r", " ")
    value = value.replace(r"\t", " ")
    return value


def extract_text_from_pdf(pdf_path: Path) -> list[str]:
    data = pdf_path.read_bytes()
    lines: list[str] = []
    stream_re = re.compile(rb"stream\r?\n(.*?)endstream", re.S)
    text_re = re.compile(r"\[(.*?)\]TJ|\((.*?)\)Tj", re.S)

    for match in stream_re.finditer(data):
        raw_stream = match.group(1)
        try:
            decoded = zlib.decompress(raw_stream)
        except Exception:
            continue
        content = decoded.decode("latin1", errors="ignore")
        for left, right in text_re.findall(content):
            chunk = left or right
            if not chunk:
                continue
            chunk = decode_pdf_string(chunk)
            chunk = re.sub(r"\d+\.?\d*", " ", chunk)
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if len(chunk) > 1:
                lines.append(chunk)
    return lines


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    pdf_path = repo / "docs" / "IEEE-CIS FRAUD DETECTION paper.pdf"
    out_path = repo / "docs" / "ieee_cis_text_extracted.txt"
    lines = extract_text_from_pdf(pdf_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path} with {len(lines)} extracted lines")


if __name__ == "__main__":
    main()
