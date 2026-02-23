import sys
from pathlib import Path
from faster_whisper import WhisperModel

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_fw.py <input_media> [model=medium] [lang=zh|auto]")
        sys.exit(1)

    inp = Path(sys.argv[1])
    model_size = sys.argv[2] if len(sys.argv) >= 3 else "medium"
    lang = sys.argv[3] if len(sys.argv) >= 4 else "zh"  # 改成 "auto" 可自动识别

    if not inp.exists():
        print(f"File not found: {inp}")
        sys.exit(1)

    out_base = inp.with_suffix("")
    out_txt = out_base.with_suffix(".txt")
    out_srt = out_base.with_suffix(".srt")
    out_vtt = out_base.with_suffix(".vtt")

    # 3070Ti：device="cuda" + compute_type="float16" 通常最快
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    segments, info = model.transcribe(
        str(inp),
        language=None if lang == "auto" else lang,
        vad_filter=True,          # 语音活动检测：减少静音段误识别
        beam_size=5,
    )

    # 写 TXT
    with out_txt.open("w", encoding="utf-8") as ftxt:
        # 写 SRT
        with out_srt.open("w", encoding="utf-8") as fsrt:
            # 写 VTT
            with out_vtt.open("w", encoding="utf-8") as fvtt:
                fvtt.write("WEBVTT\n\n")
                i = 1
                for seg in segments:
                    text = seg.text.strip()
                    if not text:
                        continue
                    ftxt.write(text + "\n")

                    start_srt = format_timestamp(seg.start, srt=True)
                    end_srt = format_timestamp(seg.end, srt=True)
                    fsrt.write(f"{i}\n{start_srt} --> {end_srt}\n{text}\n\n")

                    start_vtt = format_timestamp(seg.start, srt=False)
                    end_vtt = format_timestamp(seg.end, srt=False)
                    fvtt.write(f"{start_vtt} --> {end_vtt}\n{text}\n\n")

                    i += 1

    print(f"Done.\n- {out_txt}\n- {out_srt}\n- {out_vtt}")
    print(f"Detected language: {info.language}, probability={info.language_probability:.3f}")

if __name__ == "__main__":
    main()