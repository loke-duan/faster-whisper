import argparse
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel


NO_CLIP_TIMESTAMPS_ERROR = "No clip timestamps found"
NO_AUDIO_STREAM_ERROR = "list index out of range"

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


def transcribe_one(
    transcriber,
    media_file: Path,
    out_txt: Path,
    lang: str,
    beam_size: int,
    vad_filter: bool,
    batch_size: int,
):
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        language=None if lang == "auto" else lang,
        vad_filter=vad_filter,
        beam_size=beam_size,
        condition_on_previous_text=False,
    )

    if isinstance(transcriber, BatchedInferencePipeline):
        kwargs["batch_size"] = batch_size

    try:
        segments, info = transcriber.transcribe(str(media_file), **kwargs)
    except RuntimeError as exc:
        if NO_CLIP_TIMESTAMPS_ERROR in str(exc) and not vad_filter:
            print(
                f"Warning: {media_file} 命中 'No clip timestamps found'，自动重试并启用 VAD。"
            )
            kwargs["vad_filter"] = True
            segments, info = transcriber.transcribe(str(media_file), **kwargs)
        else:
            raise
    except IndexError as exc:
        if NO_AUDIO_STREAM_ERROR in str(exc):
            print(
                f"Warning: 跳过 {media_file}，未检测到可解码音频流（可能无音轨或媒体已损坏）。"
            )
            return False
        raise

    with out_txt.open("w", encoding="utf-8") as ftxt:
        for seg in segments:
            text = seg.text.strip()
            if text:
                ftxt.write(text + "\n")

    print(f"Done.\n- {out_txt}\n")
    print(
        f"Detected language: {info.language}, probability={info.language_probability:.3f}"
    )
    return True


def list_media_files(input_dir: Path):
    return [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe media files with faster-whisper."
    )
    parser.add_argument("model_size", nargs="?", default="medium")
    parser.add_argument("lang", nargs="?", default="zh")
    parser.add_argument("--input-dir", type=Path, default=Path("./in"))
    parser.add_argument("--output-dir", type=Path, default=Path("./out"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="cuda 推荐 float16；cpu 可选 int8/int8_float32/float32",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="追求速度时建议 1，精度优先可设为 5",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="启用 VAD，可提升静音场景稳定性，但略降速度",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="仅在 CUDA batched pipeline 下生效，可提升 GPU 利用率",
    )
    parser.add_argument(
        "--disable-batched",
        action="store_true",
        help="禁用 batched pipeline（排查兼容性时使用）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)

    media_files = list_media_files(input_dir)
    if not media_files:
        print(f"No media files found in: {input_dir}")
        raise SystemExit(0)

    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    transcriber = model
    if args.device == "cuda" and not args.disable_batched:
        transcriber = BatchedInferencePipeline(model=model)

    print(
        f"Config => model={args.model_size}, lang={args.lang}, device={args.device}, "
        f"compute_type={args.compute_type}, beam={args.beam_size}, vad={args.vad_filter}, "
        f"batch_size={args.batch_size}, batched={isinstance(transcriber, BatchedInferencePipeline)}"
    )

    success_count = 0
    skipped_count = 0

    for media_file in media_files:
        rel_path = media_file.relative_to(input_dir)
        out_txt = (output_dir / rel_path).with_suffix(".txt")
        ok = transcribe_one(
            transcriber,
            media_file,
            out_txt,
            args.lang,
            args.beam_size,
            args.vad_filter,
            args.batch_size,
        )
        if ok:
            success_count += 1
        else:
            skipped_count += 1

    print(f"Finished. success={success_count}, skipped={skipped_count}, total={len(media_files)}")


if __name__ == "__main__":
    main()
