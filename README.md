提取视频音频流中的文本 输出txt 可以实现字幕输出但没有实现

环境Python 3.10.11

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

需验证：

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

pip install faster-whisper
