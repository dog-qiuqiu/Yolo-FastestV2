# 训练权重说明

> 本目录用于存放训练得到的模型权重文件 (`*.pth`)。  
> 权重文件本身不纳入 Git 版本控制（已在 `.gitignore` 排除），仅本 README 入库以记录训练成绩。

## 文件命名规则

```
{model_name}-{epoch}-epoch-{mAP@0.5}ap-model.pth
```

例如：`dms-kaggle-150-epoch-0.699052ap-model.pth` 表示在 `dms-kaggle` 数据集上训练 150 epoch、验证集 mAP@0.5 = 69.91% 的权重。

---

## DMS Kaggle 训练记录

- **训练时间**：2026-04-28
- **训练配置**：`@/data/dms-kaggle.data`
- **训练命令**：`python train.py --data data/dms-kaggle.data`
- **数据集**：DMS Kaggle (5 类：Open Eye / Closed Eye / Cigarette / Phone / Seatbelt)
- **训练集 / 验证集**：5957 / 2389 张
- **输入分辨率**：352 × 352
- **Batch Size**：64
- **优化器**：SGD (momentum=0.949, weight_decay=0.0005)
- **学习率策略**：MultiStepLR，初始 0.001，在 epoch 150 / 250 各衰减 10x
- **总 epoch**：300
- **模型参数量**：239,080（轻量级）
- **环境**：Python 3.10 + PyTorch 2.5.1+cu121 (conda env `DMS`)

### mAP@0.5 演化表

| Epoch | LR       | mAP@0.5    | 备注                |
|------:|:---------|:----------:|:--------------------|
|    10 | 0.001    | 55.83%     | warmup 后首次评估   |
|    20 | 0.001    | 59.77%     |                     |
|    30 | 0.001    | 65.38%     |                     |
|    40 | 0.001    | 65.08%     |                     |
|    50 | 0.001    | 65.98%     |                     |
|    60 | 0.001    | 68.96%     |                     |
|    70 | 0.001    | 68.60%     |                     |
|    80 | 0.001    | 69.72%     | LR=0.001 阶段最佳   |
|    90 | 0.001    | 69.69%     |                     |
|   100 | 0.001    | 68.65%     |                     |
|   110 | 0.001    | 68.20%     |                     |
|   120 | 0.001    | 68.19%     |                     |
|   130 | 0.001    | 69.13%     |                     |
|   140 | 0.001    | 69.44%     |                     |
| **150** | **0.001→0.0001** | **69.91%** | ⭐ **历史最佳** |
|   160 | 0.0001   | 69.58%     |                     |
|   170 | 0.0001   | 69.57%     |                     |
|   180 | 0.0001   | 69.23%     |                     |
|   190 | 0.0001   | 69.58%     |                     |
|   200 | 0.0001   | 69.20%     |                     |
|   210 | 0.0001   | 69.07%     |                     |
|   220 | 0.0001   | 69.17%     |                     |
|   230 | 0.0001   | 69.23%     |                     |
|   240 | 0.0001   | 69.05%     |                     |
|   250 | 0.0001→0.00001 | 69.00% | 第二次 LR 衰减     |
|   260 | 0.00001  | 68.80%     |                     |
|   270 | 0.00001  | 69.08%     |                     |
|   280 | 0.00001  | 68.81%     |                     |
|   290 | 0.00001  | 68.74%     |                     |
|   300 | 0.00001  | —          | 训练结束（不保存）  |

> 训练脚本仅在 `epoch % 10 == 0 and epoch > 0` 时评估并保存，因此最后一个 epoch（299）未单独保存权重。

### 推荐权重

- **🏆 最佳精度**：`dms-kaggle-150-epoch-0.699052ap-model.pth` （**mAP@0.5 = 69.91%**）
- **📦 备选稳定权重**：`dms-kaggle-80-epoch-0.697151ap-model.pth` （第一次 LR 衰减前的高点，泛化性较好）

### 推理 / 评估

```bash
# 推理单张图
python test.py --data data/dms-kaggle.data --weights weights/dms-kaggle-150-epoch-0.699052ap-model.pth --img <path-to-image>

# 在验证集上评估
python evaluation.py --data data/dms-kaggle.data --weights weights/dms-kaggle-150-epoch-0.699052ap-model.pth

# 导出 ONNX
python pytorch2onnx.py --data data/dms-kaggle.data --weights weights/dms-kaggle-150-epoch-0.699052ap-model.pth
```

### 训练观察

- **收敛曲线**：Total Loss 从 ~30 → ~0.74，CIoU/Obj/Cls 三项均稳定下降
- **mAP 平台**：在 LR=0.001 阶段约 80~90 epoch 已接近瓶颈，后续 LR 衰减仅微幅提升
- **过拟合风险**：Epoch 150 之后 mAP 在 ±0.5% 区间震荡，未出现明显下滑，模型未过拟合
- **复盘建议**：若资源充裕，可尝试 (1) 提升输入分辨率到 416/512； (2) 启用 datasets.py 中已注释的更多数据增强 (motion_blur / random_resize / augment_hsv)； (3) 调小 `steps` 让 LR 更早衰减

---

## 历史训练记录

> 后续每次重要训练完成后，请在此追加一节 `## <模型名> 训练记录`，至少包含：训练时间、配置文件、命令、最佳权重、mAP 表格。


---

## ONNX / NCNN 部署说明 (2026-04-28)

### 模型转换

1. **PyTorch → ONNX** (使用 `pytorch2onnx.py`):
   ```bash
   python pytorch2onnx.py \
     --data data/dms-kaggle.data \
     --weights weights/dms-kaggle-150-epoch-0.699052ap-model.pth \
     --output weights/dms-kaggle.onnx \
     --simplify
   ```
   产出: `weights/dms-kaggle.onnx` (~968 KB) 与 `weights/dms-kaggle-sim.onnx` (~982 KB, 经 onnxsim 简化, 推荐用于下游)

2. **ONNX → NCNN** (使用 `pnnx`):
   ```bash
   pnnx weights/dms-kaggle-sim.onnx inputshape=[1,3,352,352]
   ```
   产出 (FP16 量化, 默认):
   - `weights/dms_kaggle_sim.ncnn.param` (~13 KB) — 网络结构
   - `weights/dms_kaggle_sim.ncnn.bin` (~478 KB) — 权重 (FP16)
   - `weights/dms_kaggle_sim_ncnn.py` — 自动生成的推理样板

   如需 FP32: `pnnx ... fp16=0` (.bin 变为 ~940 KB)

### 模型输出格式 (`detector.export_onnx=True` 分支)

- 两个尺度: `(1, 22, 22, 20)` (stride=16) 与 `(1, 11, 11, 20)` (stride=32)
- 每 cell 的 20 通道 (NHWC):
  - `[0:12]` per-anchor 回归 (cx,cy,w,h) × 3 anchors, 已 sigmoid
  - `[12:15]` per-anchor obj (1) × 3 anchors, 已 sigmoid
  - `[15:20]` 共享 cls (5 类), 已 softmax

### 推理脚本

- **`infer_onnx.py`** — 用 onnxruntime 推理 (CPU/GPU 自动选择)
- **`infer_ncnn.py`** — 用 ncnn-python 推理

```bash
python infer_onnx.py --data data/dms-kaggle.data \
    --onnx weights/dms-kaggle-sim.onnx \
    --img <img> --conf 0.3 --iou 0.4 --bench 50

python infer_ncnn.py --data data/dms-kaggle.data \
    --param weights/dms_kaggle_sim.ncnn.param \
    --bin weights/dms_kaggle_sim.ncnn.bin \
    --img <img> --conf 0.3 --iou 0.4 --bench 50
```

### x86 CPU 性能 (i9, 单线程参考)

| Backend       | Latency | FPS    |
|---------------|---------|--------|
| ONNX Runtime  | 3.4 ms  | 290    |
| NCNN (FP16)   | 4.7 ms  | 215    |

> NCNN 在 x86 略慢于 ONNX (FP16 ↔ FP32 转换开销); 部署到 ARM/嵌入式平台时 NCNN 才显出优势 (体积小、编译简单、对低端 CPU 有针对性优化)。

### Python 依赖

- `onnx==1.21.0`, `onnxruntime==1.18.1`, `onnxsim==0.6.2`
- `ncnn==1.0.20260114` (`pip install ncnn`)
- `pnnx` (`pip install pnnx`, 提供 `pnnx.exe` 转换工具)
- 注意: 安装 `ncnn` 会拉 numpy 2.x; 需手动 `pip install "numpy<2"` 防 onnxruntime 失效

### 已知坑

- `cv2.transpose(2,0,1)` 后 numpy 数组**非连续**, 直接传 `ncnn.Mat()` 会读乱内存导致输出错乱 (obj 通道全 0). 必须 `np.ascontiguousarray()` 先连续化.
- ncnn `Mat` 引用的 numpy 数组**必须保持作用域**, 否则段错误 (Windows: 退出码 -1073741819 / `0xC0000005`).
- pnnx 默认开启 fp16; 大多数情况无影响, 极少数对边界值敏感的层可换 `fp16=0`.
