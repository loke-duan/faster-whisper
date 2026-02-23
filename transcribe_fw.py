import sys
from pathlib import Path
from faster_whisper import WhisperModel

MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".flv",
}

def format_timestamp(seconds: float, srt=False) -> str:
    # seconds -> HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)
    ms = int(round(seconds * 1000.0))
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def transcribe_one(model: WhisperModel, media_file: Path, out_txt: Path, lang: str):
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # 3070Ti：device="cuda" + compute_type="float16" 通常最快
    segments, info = model.transcribe(
        str(media_file),
        language=None if lang == "auto" else lang,
        vad_filter=True,          # 语音活动检测：减少静音段误识别
        beam_size=5,
    )

    # 写 TXT
    with out_txt.open("w", encoding="utf-8") as ftxt:
        i = 1
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            ftxt.write(text + "\n")
            i += 1

    print(f"Done.\n- {out_txt}\n")
    print(f"Detected language: {info.language}, probability={info.language_probability:.3f}")


def list_media_files(input_dir: Path):
    return [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]


def main():
    input_dir = Path("E:/PycharmProjects/faster-whisper/in").expanduser()
    output_dir = Path("E:/PycharmProjects/faster-whisper/out").expanduser()
    model_size = sys.argv[1] if len(sys.argv) >= 2 else "medium"
    lang = sys.argv[2] if len(sys.argv) >= 3 else "zh"  # 改成 "auto" 可自动识别

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    media_files = list_media_files(input_dir)
    if not media_files:
        print(f"No media files found in: {input_dir}")
        sys.exit(0)

    # 3070Ti：device="cuda" + compute_type="float16" 通常最快
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    for media_file in media_files:
        rel_path = media_file.relative_to(input_dir)
        out_txt = (output_dir / rel_path).with_suffix(".txt")
        transcribe_one(model, media_file, out_txt, lang)

if __name__ == "__main__":
    main()
