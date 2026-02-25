提取视频音频流中的文本，支持输出 `txt` 或 `srt` 字幕文件。

环境 Python 3.10.11

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper
```

需验证：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

## 使用示例

```bash
python transcribe_fw.py medium zh --input-dir ./in --output-dir ./out --output-format srt
```

`--output-format` 可选：
- `txt`：仅输出文本（默认）
- `srt`：仅输出字幕
- `both`：同时输出 txt 与 srt
