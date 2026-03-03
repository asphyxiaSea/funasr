from funasr import AutoModel
import torch.nn.functional as F
import numpy as np
import torch
import base64

model = AutoModel(model="models/SPKmodels/cam++")

# 1. Generate speaker1 embedding
speaker1_emb1 = model.generate(input="models/SPKmodels/cam++/examples/speaker1_a_cn_16k.wav")[0]["spk_embedding"]

speaker1_emb1 = F.normalize(speaker1_emb1, p=2, dim=1)

# 将向量转换为二进制数据并转换为Base64字符串
vector1  = speaker1_emb1.detach().cpu().numpy().astype(np.float32)
base64_str1 = base64.b64encode(vector1.tobytes()).decode("utf-8")




# # 将Base64字符串解码回二进制数据，并转换为原始向量
# base64_str1 = base64.b64decode(base64_str1)
# vector1  = np.frombuffer(base64_str1, dtype=np.float32)
# vector1  = torch.tensor(vector1).unsqueeze(0)


# speaker1_emb2 = model.generate(input="models/SPKmodels/cam++/examples/speaker1_b_cn_16k.wav")[0]["spk_embedding"]

# speaker1_emb2 = F.normalize(speaker1_emb2, p=2, dim=1)
# vector2 = speaker1_emb2.detach().cpu().numpy().astype(np.float32)
# binary2 = vector2.tobytes()
# base64_str2 = base64.b64encode(binary2).decode("utf-8")

# base64_str2 = base64.b64decode(base64_str2)
# vector2 = np.frombuffer(base64_str2, dtype=np.float32)
# vector2 = torch.tensor(vector2).unsqueeze(0)


# # 相似度
# score = F.cosine_similarity(vector1, vector2, dim=1)
# print("相似度:", score.item())
