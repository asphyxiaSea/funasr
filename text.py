# from funasr import AutoModel
# import torch.nn.functional as F

# model = AutoModel(model="models/SPKmodels/cam++")

# # 1. Generate speaker1 embedding
# speaker1_emb1 = model.generate(input="models/SPKmodels/cam++/examples/speaker1_a_cn_16k.wav")[0]["spk_embedding"]
# speaker1_emb2 = model.generate(input="models/SPKmodels/cam++/examples/speaker1_b_cn_16k.wav")[0]["spk_embedding"]
# speaker2_emb1 = model.generate(input="models/SPKmodels/cam++/examples/speaker2_a_cn_16k.wav")[0]["spk_embedding"]

# # 相似度
# score = F.cosine_similarity(speaker1_emb1, speaker2_emb1)

# print("相似度:", score.item())

# model = AutoModel(model="models/VADmodels")



from funasr import AutoModel

wav_file = "models/VADmodels/speech_fsmn_vad_zh/example/vad_example.wav"

model = AutoModel(model="models/VADmodels/speech_fsmn_vad_zh")

res = model.generate(input=wav_file)
print(res)


import soundfile
import os

speech, sample_rate = soundfile.read(wav_file)

chunk_size = 200  # ms
chunk_stride = int(chunk_size * sample_rate / 1000)

cache = {}

total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
for i in range(total_chunk_num):
    speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(
        input=speech_chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        disable_pbar=True,
    )
    # print(res)
    if len(res[0]["value"]):
        print(res)