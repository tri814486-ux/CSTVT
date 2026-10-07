# 四模型单模块消融

`manuscript_four_ablation_model.py` 以 `manuscript_primary_model.py` 为唯一实现依据，先构建完整模型，再删除指定模块。

```python
from manuscript_four_ablation_model import build_manuscript_ablation

model = build_manuscript_ablation(
    "no_cpe", seed=42, image_size=160, channels=(16, 32, 64, 128),
    depth=2, heads=4, grid_side=10, iterations=2, expansion=2, num_classes=5,
)
```

| key | 模型 | 唯一变化 |
| --- | --- | --- |
| `full` | CSTVT | 保留原模型全部模块 |
| `vit` | CNN-ViT | 取消 STS 与 TU 映射；Class Token 与全部归一化视觉 Token 直接进行 MHSA |
| `no_cpe` | CSTVT 去 CPE | 跳过 CPE 残差分支；绝对位置编码 P 保留 |
| `no_dwconv` | CSTVT 去 DWConv | 仅将 ConvFFN 中 3×3 depthwise 卷积换为 Identity；CPE 及其卷积保留 |

四组均保留 CNN 特征提取、Class Token、绝对位置编码 P、MHSA 和未被指定删除的 ConvFFN 步骤。相同 seed 下所有保留参数初始化逐元素相同。`model.ablation_metadata` 记录删改与保留内容。`seed=` 仅控制临时 CPU 构造随机状态，不替代训练数据顺序、数据增强或 CUDA 随机种子设置。

## 原实现语义限制

原实现每个 block 在 ConvFFN 前更新 Class Token，分类器读取最后一个 block 的 Class Token。因此，最后一个 block 的 ConvFFN 输出不会影响分类 logits；先前 block 的 ConvFFN 会通过后续 block 影响分类。本消融保持这一既有语义，不在消融过程中改变完整模型。深度为 1 时，去 DWConv 不会改变分类前向结果；本四组对照建议沿用原来的深度 2，并在分析模块贡献时如实说明此限制。

不预设四组准确率或损失的排名。模块删减结果须由相同训练和评价条件下的真实记录确定。

## CPU 验证

```powershell
python test_manuscript_four_ablation_model.py
```

检查原源码 SHA256、共享参数初始化、完整模型前向完全一致、CPE 删除不将输入翻倍、DWConv 只删除 FFN 指定步骤、直接视觉 Token 注意力、输出和有效梯度有限、原最后 FFN 梯度语义及 train/eval 模式。
