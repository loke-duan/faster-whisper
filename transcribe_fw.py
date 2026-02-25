import argparse
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel


# 预留：上游在某些边界音频中可能抛出的时间戳错误文案（当前脚本未直接使用）
NO_CLIP_TIMESTAMPS_ERROR = "No clip timestamps found"
# 当前已知：当媒体无可解码音频流时，底层可能抛出的 IndexError 文案
NO_AUDIO_STREAM_ERROR = "list index out of range"

# 支持递归扫描的媒体文件扩展名集合（统一使用小写做匹配）
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


def format_srt_timestamp(seconds: float) -> str:
    """将秒数格式化为 SRT 时间戳（HH:MM:SS,mmm）。"""
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_one(
    transcriber,
    media_file: Path,
    out_txt: Path,
    output_format: str,
    lang: str,
    beam_size: int,
    vad_filter: bool,
    batch_size: int,
):
    """转写单个媒体文件并写入同名 txt/srt。

    返回值：
    - True: 成功转写并输出文本。
    - False: 因无音轨/音频损坏等可预期问题被跳过。
    """
    # 确保输出目录存在，避免写入文本时因目录缺失报错
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # 构建 faster-whisper 推理参数：auto 语言时传入 None 让模型自动检测
    # 说明：condition_on_previous_text=False 可减少跨段“错误延续”
    kwargs = dict(
        language=None if lang == "auto" else lang,
        vad_filter=vad_filter,
        beam_size=beam_size,
        condition_on_previous_text=False,
    )

    # 仅 BatchedInferencePipeline 支持 batch_size 参数
    if isinstance(transcriber, BatchedInferencePipeline):
        kwargs["batch_size"] = batch_size

    try:
        # 将 Path 转成字符串，兼容底层 API 的路径参数类型
        segments, info = transcriber.transcribe(str(media_file), **kwargs)
    except IndexError as exc:
        # 某些损坏文件或无音轨媒体会触发该错误，按“跳过”处理
        if NO_AUDIO_STREAM_ERROR in str(exc):
            print(
                f"Warning: 跳过 {media_file}，未检测到可解码音频流（可能无音轨或媒体已损坏）。"
            )
            return False
        # 不是已知可跳过异常时，继续向上抛出，便于尽早暴露问题
        raise

    # segments 是生成器，若同时输出 txt/srt 需先物化，避免二次遍历为空
    segments = list(segments)

    # 使用 UTF-8 输出，避免中文/多语言内容写入时出现编码问题
    if output_format in {"txt", "both"}:
        with out_txt.open("w", encoding="utf-8") as ftxt:
            # 逐段写入非空文本，每行一个 segment
            for seg in segments:
                text = seg.text.strip()
                if text:
                    ftxt.write(text + "\n")

    if output_format in {"srt", "both"}:
        out_srt = out_txt.with_suffix(".srt")
        with out_srt.open("w", encoding="utf-8") as fsrt:
            subtitle_index = 1
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue

                fsrt.write(f"{subtitle_index}\n")
                fsrt.write(
                    f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}\n"
                )
                fsrt.write(f"{text}\n\n")
                subtitle_index += 1

    # 打印输出文件位置，方便批处理时快速定位结果
    if output_format == "txt":
        print(f"Done.\n- {out_txt}\n")
    elif output_format == "srt":
        print(f"Done.\n- {out_txt.with_suffix('.srt')}\n")
    else:
        print(f"Done.\n- {out_txt}\n- {out_txt.with_suffix('.srt')}\n")

    # 输出语言识别结果与置信度，便于人工核对 auto 识别效果
    print(
        f"Detected language: {info.language}, probability={info.language_probability:.3f}"
    )
    return True


def list_media_files(input_dir: Path):
    """递归收集输入目录下所有支持的媒体文件路径。"""
    # 递归收集目录内所有受支持的媒体文件
    return [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]


def parse_args():
    """定义并解析命令行参数。"""
    # 这里保留了位置参数与可选参数的混合形式，以兼容既有调用习惯
    parser = argparse.ArgumentParser(
        description="Transcribe media files with faster-whisper."
    )
    # 位置参数：模型大小（如 tiny/base/small/medium/large）
    parser.add_argument("model_size", nargs="?", default="medium")
    # 位置参数：语言代码；传 auto 时会启用自动语言检测
    parser.add_argument("lang", nargs="?", default="zh")

    # 可提升静音场景稳定性，但略降速度
    parser.add_argument("--vad-filter", dest="vad_filter", action="store_true", help="启用 VAD")
    parser.add_argument("--no-vad-filter", dest="vad_filter", action="store_false", help="禁用 VAD")
    parser.set_defaults(vad_filter=True)  # 默认开VAD就设True；否则False

    # 输入/输出默认目录为本地开发路径，可按需在命令行覆盖
    parser.add_argument("--input-dir", type=Path, default=Path("E:/PycharmProjects/faster-whisper/in"))
    parser.add_argument("--output-dir", type=Path, default=Path("E:/PycharmProjects/faster-whisper/out"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--output-format",
        default="txt",
        choices=["txt", "srt", "both"],
        help="输出格式：txt、srt 或 both（同时输出）",
    )

    parser.add_argument(
        "--compute-type",
        default="float16",
        help="cuda 推荐 float16；cpu 可选 int8/int8_float32/float32",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="追求速度时建议 1，精度优先可设为 5",
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
    """脚本入口：加载参数 -> 扫描媒体 -> 批量转写 -> 汇总结果。"""
    args = parse_args()
    # 展开 ~，允许通过用户目录的相对写法传参
    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    # 输入目录必须存在且是目录；否则直接退出并返回错误码 1
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)

    # 先一次性收集待处理文件，便于后续统计 total 数量
    media_files = list_media_files(input_dir)
    # 无文件可处理时按“正常结束”退出（状态码 0）
    if not media_files:
        print(f"No media files found in: {input_dir}")
        raise SystemExit(0)

    # 初始化 Whisper 模型（设备与精度由命令行参数控制）
    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    # 默认使用普通推理；在 CUDA 且未禁用时切换到 batched pipeline
    transcriber = model
    if args.device == "cuda" and not args.disable_batched:
        transcriber = BatchedInferencePipeline(model=model)

    # 输出最终生效配置，便于复现实验与排查问题
    print(
        f"Config => model={args.model_size}, lang={args.lang}, device={args.device}, "
        f"compute_type={args.compute_type}, beam={args.beam_size}, vad={args.vad_filter}, "
        f"batch_size={args.batch_size}, output_format={args.output_format}, "
        f"batched={isinstance(transcriber, BatchedInferencePipeline)}"
    )

    # 统计：成功转写数量与跳过数量（如无音轨文件）
    success_count = 0
    skipped_count = 0

    # 逐文件处理，互不影响：单个文件异常/跳过不阻塞整体批处理
    for media_file in media_files:
        # 保持输入目录层级结构，仅将后缀统一转换为 .txt
        rel_path = media_file.relative_to(input_dir)
        out_txt = (output_dir / rel_path).with_suffix(".txt")
        ok = transcribe_one(
            transcriber,
            media_file,
            out_txt,
            args.output_format,
            args.lang,
            args.beam_size,
            args.vad_filter,
            args.batch_size,
        )
        if ok:
            success_count += 1
        else:
            skipped_count += 1

    # 打印批处理汇总，便于快速评估本次任务完成情况
    print(f"Finished. success={success_count}, skipped={skipped_count}, total={len(media_files)}")


if __name__ == "__main__":
    main()
