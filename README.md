# CSTVT
CSTVT 是一个融合卷积神经网络（CNN）与 Super Token Transformer 的视觉模型。CNN 分支用于提取局部纹理、边缘和细节特征，Super Token Transformer 分支通过 Super Token 机制对 token 进行聚合与压缩，在减少自注意力计算量的同时建模全局依赖，从而在 [图像分类/目标检测/分割/视频理解等任务] 上兼顾表达能力与计算效率。
