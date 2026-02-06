from typing import Any, Dict, cast
from pathlib import Path
from funasr import AutoModel
import sys

BASE_DIR = Path(__file__).resolve().parent
FUNASR_DIR = BASE_DIR / "FunASR"   # 根据你的实际目录调整

sys.path.append(str(FUNASR_DIR))

from FunASR.model import FunASRNano

# 直接调用ASR模型
def main1():
    model_dir = "models/ASRmodels/Fun-ASR-Nano-2512"
    wav_path = "models/ASRmodels/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"
    
    m, kwargs = cast(
        tuple[FunASRNano, Dict[str, Any]],
        FunASRNano.from_pretrained(model=model_dir, device="cuda:0"),
    )
    m.eval()

    res = m.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]["text"]
    return text



# 使用AutoModel加载ASR+VAD模型
def main2():
    model_dir = "models/ASRmodels/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_model_dir = "models/VADmodels/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    wav_path = "models/ASRmodels/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"
    model = AutoModel(
        model=model_dir,
        vad_model=vad_model_dir,
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
        disable_update = True,
    )
    
    res = model.generate(input=[wav_path], cache={}, batch_size=1)
    text = res[0]["text"]
    return text

main1()