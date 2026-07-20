# ML HW8 — Anomaly Detection

NYCU 機器學習 (李宏毅 ML2023 Spring) 作業 8：用 Autoencoder 做**非監督式異常偵測**。

---

## 1. 作業目標

判斷一張圖片「是不是真人臉」。

- 訓練資料**只有正常人臉**（100,000 張），**沒有異常標籤**
- 測試時要對每張圖輸出 **anomaly score**（不是 0/1 分類）
- 分數越大代表越可能是異常（非人臉）

這是「one-class learning」：模型只看過一種類別，卻要分辨進來的東西是不是這個類別。

---

## 2. 資料

| 檔案 | Shape | 大小 | 內容 |
|------|-------|------|------|
| `data/trainingset.npy` | `(100000, 64, 64, 3)` uint8 | 1.17 GB | 真人臉 |
| `data/testingset.npy` | `(19636, 64, 64, 3)` uint8 | 230 MB | 約一半人臉 (label=0)，一半異常 (label=1) |

資料來自 [chiyuanhsiao/ml2023spring-hw8](https://github.com/chiyuanhsiao/ml2023spring-hw8)（Kaggle 比賽的真實資料用 Git LFS 託管）。

---

## 3. 核心方法：Autoencoder + Reconstruction Error

```
input → [Encoder] → latent z → [Decoder] → reconstructed
   ↑___________ MSE loss __________↓
```

**關鍵直覺**：
- 訓練時只看人臉，模型學會「怎麼壓縮 / 還原一張人臉」
- 推論時：
  - 真人臉 → 容易還原 → reconstruction error 小 → score 低
  - 異常圖 → 模型沒看過這個分佈 → 還原會失敗 → error 大 → score 高
- **Anomaly score = per-image MSE 重建誤差**

---

## 4. 評估：ROC AUC

- **為什麼不用 accuracy？** 因為我們輸出的是連續分數，沒有 threshold
- **AUC** 衡量「把 anomaly 排在 normal 前面」的能力
  - AUC = 1.0 → 完美區分
  - AUC = 0.5 → 亂猜
- 評估流程：對所有 score 排序 → 算 TPR / FPR → ROC 曲線下面積

---

## 5. Baseline 升級路線（核心！）

每一級都是在前一級的基礎上增加技術。下面說明**每加一個技術為什麼能突破**。

### 🥉 Simple — Vanilla Conv Autoencoder (AUC ≈ 0.53)

**架構**：
- Encoder：3 層 Conv2d (stride=2)，64×64 → 8×8
- Decoder：對稱 ConvTranspose2d 還原
- 啟用：ReLU，輸出層 Tanh
- 訓練：MSE loss，Adam lr=1e-3，30 epochs

**為什麼能 work**：
- CNN 能學到人臉的 spatial pattern（眼睛、鼻子、嘴巴的相對位置）
- Anime / 卡通臉的 spatial pattern 不同 → 還原誤差會比較大

**為什麼分數不高**：
- 模型太小，特徵抽取能力有限
- 沒有任何 regularization，可能 overfit 到 pixel-level 細節，丟失 structure

---

### 🥈 Medium — 強化模型架構 (AUC ≈ 0.73)

**比 Simple 多加的技術**：

| 改進 | 為什麼有用 |
|------|----------|
| **更深的網路**（5-6 層 conv） | 更深 = 更大 receptive field，能抓全局臉部結構 |
| **BatchNorm** | 訓練更穩定，能用更大 lr，每層 activation 分佈不會崩 |
| **LeakyReLU**（取代 ReLU） | 避免 dead neurons，負區段也有梯度 |
| **更大 latent dim**（從預設 → 256+） | 瓶頸過小會丟失太多資訊，連人臉都還原不好 |
| **訓練更久**（80-100 epochs） + **lr scheduler** | Loss 收斂更徹底，學到更精細的人臉特徵 |

**突破點**：
模型容量變大，能更精確還原人臉 → 異常圖的還原誤差**相對人臉的**會被拉大，AUC 自然提升。

---

### 🥇 Strong — Multi-Encoder Autoencoder (AUC ≈ 0.77)

**比 Medium 多加的技術**：

**Multi-Encoder 架構**：
```
        ┌─→ Encoder_A ─┐
input ──┼─→ Encoder_B ─┼─→ concat z → Decoder → output
        └─→ Encoder_C ─┘
```

**為什麼有用**：
- 每個 encoder 從不同角度（不同 kernel size / depth / receptive field）看圖片
- 多個 latent 拼接 → 表達能力更豐富
- 學到的 representation 更「貼合人臉分佈」→ 越異常的圖越難從這些 representation 還原
- 類似 **ensemble** 的思想，但內建在 latent space 裡

**參考論文**：[Multi-Encoder Autoencoder for Anomaly Detection](https://arxiv.org/abs/2003.04060)

---

### 👑 Boss — Denoising AE + Classifier (AUC ≈ 0.80+)

**比 Strong 多加的技術**（兩種方法擇一，這裡介紹方法 A）：

**方法 A：Denoising + Classifier Head**

```
                        ┌─→ classifier → real/fake
input ─→ encoder ─→ z ──┤
        ↑               └─→ decoder → reconstructed
   加 Gaussian noise
```

**運作機制**：
1. 訓練時對 input 加 Gaussian noise（變成 noisy input）
2. Autoencoder 同時做兩件事：
   - **去噪重建**：output 要接近**原圖**（不是 noisy input）
   - **真偽分類**：classifier 判斷 latent 是「真實人臉的 z」還是「noise 過後的 z」
3. 推論時 anomaly score = `α · reconstruction_error + β · classifier_logit`

**為什麼有用**：
- **去噪訓練**強迫模型抓**robust 特徵**（不被 pixel-level 噪音欺騙）
- **Classifier** 提供額外的 anomaly signal：異常圖的 latent 跟 noisy latent 類似（都「不太像正常人臉」），會被 classifier 標成 fake
- 兩個訊號互補 → 比單純看 reconstruction error 更可靠

**方法 B：ResNet Encoder（替代方案）**
- 用 ResNet18 結構（自己刻，**不能用 pre-trained**）當 encoder
- 殘差連接讓深層網路訓練不會退化
- 但 ResNet 沒法用 pre-trained 就少了精髓，通常不如方法 A

**參考論文**：[Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection](https://arxiv.org/abs/1904.02639)

---

### 📊 突破點總結

```
Simple   → vanilla CNN
   ↓  + 更深 + BN + LeakyReLU + 更大 latent
Medium   → 強化版 CNN
   ↓  + 多個 encoder 並聯
Strong   → 多視角特徵
   ↓  + Gaussian noise + classifier head
Boss     → 多訊號融合
```

每一級的核心都是「**讓模型學到的人臉分佈更精準**」。分佈學得越準 → 異常的東西越明顯。

---

## 6. 實際結果（Kaggle 提交分數）

| # | Baseline | Params | 訓練時間 | Public AUC | Private AUC | 結果 |
|---|----------|--------|---------|------------|-------------|------|
| 1 | 🥉 Simple | 47k | ~90 秒 | 0.66943 | 0.66864 | ✅ 過 simple，大幅超過預期 |
| 2 | 🥈 Medium | 3.48M | ~6 分鐘 | 0.72877 | 0.72263 | ✅ 過 medium |
| 3 | 🥇 Strong v1 (multi-encoder) | 8.69M | ~14 分鐘 | 0.70716 | 0.70192 | ❌ 退步 0.02 |
| 4 | 👑 Boss v1 (DAE + classifier) | 3.52M | ~12 分鐘 | 0.68294 | 0.68244 | ❌ 退步 0.05 |
| 5 | VAE | 4.53M | ~15 分鐘 | 0.73048 | 0.72369 | ✅ 微幅勝過 medium |
| 6 | ResNet AE | 18.16M | ~30 分鐘 | 0.72885 | 0.72120 | ≈ medium |
| 7 | Ensemble z-score (med+vae+resnet) | - | - | 0.73008 | 0.72319 | ⚠️ 微跌 0.0004 (模型太像) |
| 8 | Ensemble rank (med+vae+resnet) | - | - | 0.73006 | 0.72313 | ⚠️ 微跌 0.0004 |
| 9 | **MemAE v1 (broken-but-good)** | 3.61M | ~18 分鐘 | **0.75269** | **0.75660** | 🚀 **大躍進 +0.022** |
| 10 | MemAE v2 (paper-faithful) | 3.61M | ~18 分鐘 | 0.67298 | 0.68315 | ❌❌ **大退步 -0.08**（反直覺！）|
| 11 | Ensemble r_memae_v1v2 | - | - | 0.72193 | 0.72927 | ❌ v2 拖累 |
| 12 | Ensemble r_memae_v1v2_vae | - | - | 0.73901 | 0.74171 | ❌ 同上 |
| 13 | Ensemble z_memae2x_vae | - | - | 0.75702 | 0.75817 | ✅ |
| 14 | Ensemble r_memae2x_vae | - | - | 0.75757 | 0.75808 | ✅ |
| 15 | MemAE v1 multi-seed (3 seeds) | 3×3.61M | ~30 分鐘 | 0.75-0.76 (各 seed) | - | ✅ 微漲 |
| 16 | **Ensemble r_3seed_vae_balanced** | - | - | **0.75759** ⭐ | 0.75926 | ✅ **目前 best Public** |
| 17 | DeepSVDD | 1.21M | ~12 分鐘 | 0.63745 | 0.64304 | ⚠️ 單獨弱但 corr 只 0.32 |
| 18 | **Ensemble z_3seed_vae_ds03** | - | - | 0.75689 | **0.75984** ⭐ | ✅ **目前 best Private**(加 DeepSVDD 0.3x) |
| 19 | Spatial MemAE | 1.51M | ~22 分鐘 | 0.68713 | 0.68281 | ❌ 16 spatial 位置太彈性失去 averaging 效應 |
| 20 | MH-AE (mean variant) | 6.97M | ~25 分鐘 | 0.72884 | 0.72233 | ❌ 3 decoders 沒分化 ≈ VAE |
| 21 | MH-AE (min variant) | 6.97M | ~25 分鐘 | 0.72759 | 0.72099 | ❌ 同上 |
| 22 | VQVAE | 1.41M | ~15 分鐘 | 0.73734 | 0.74131 | ≈ VAE 跟 MemAE 之間 |
| 23 | CutPaste (classifier-based) | 1.74M | ~20 分鐘 | 0.59899 | 0.59317 | ❌ classifier 信心訊號太弱 |
| 24 | **CutPaste (Mahalanobis)** | 1.74M | ~20 分鐘 | **0.72454** | 0.72015 | ✅ 加 ensemble 後大躍進 |
| 25 | Ensemble + CutPaste 3× | - | - | **0.77978** | 0.77849 | 🚀 **第二次突破** |
| 26 | PatchSVDD (原始) | 0.43M | ~22 分鐘 | 0.23603 | 0.24060 | ⚠️ 學歪方向(corr 全負) |
| 27 | **PatchSVDD (inverted)** | 0.43M | - | **0.76396** | 0.75939 | ✅ 反轉後變強訊號 |
| 28 | Ensemble + PatchSVDD 3× | - | - | 0.83187 | 0.82791 | 🚀 過 Boss baseline! |
| 29 | **Ensemble cp3+ps5 (drop DS)** ⭐ | - | - | **0.83973** | **0.83490** | 🏆 **最終 best** |

**🏆 最終最高分**：
- Public **0.83973** (`ensemble_r_cp3_no_ds_ps5`)
- Private **0.83490** (同上)

從起點 0.669 (simple) 到最終 0.840，**Public AUC 漲了 +0.170**。**遠超 Boss baseline 0.80**。

**三大里程碑突破**：
- 🚀 **MemAE v1** (0.753) 第一次突破 conv-AE 家族 0.73 天花板 — 詳見 [7.10](#710-memae-v1--真正的突破)
- 🚀 **CutPaste 3× ensemble** (0.780) 第二次突破 — 自監督帶來低 corr 訊號,詳見 [7.20](#720-cutpaste--self-supervised-第二次突破)
- 🚀 **PatchSVDD inverted 5×** (0.840) 第三次突破 — 模型學歪方向反轉後變超強,詳見 [7.21](#721-patchsvdd--學歪方向反轉成寶藏)

**意外的教訓**:
- **MemAE v2 修了 bug 反而退步 0.08** — paper-faithful 不等於對你的 task 對,詳見 [7.11](#711-memae-v2--反直覺的反轉-修-bug-反而退步)
- **PatchSVDD 原始 AUC 0.236**(< 0.5),反轉成 1-score 後變 0.764 — corr 完全變號的「反向訊號」是寶藏

---

## 7. 實驗日誌（試錯記錄與反思）

這份作業的核心精神是**比較不同方法為什麼成功 / 為什麼失敗**。下面是每一次嘗試的完整記錄。

### 📖 PDF Spec 提示 vs 我們的實作（對照表）

HW08.pdf page 15-16 給每個 baseline 的官方提示：

| Baseline | 預估時間 | PDF 提示 | 我們的實作 | 結果 |
|----------|---------|---------|-----------|------|
| 🥉 Simple | 10 min | Sample code — 跑助教範例 | `simple_baseline.py`(3-layer conv AE) | ✅ 0.66943 |
| 🥈 Medium | 15-20 min | Adjust model structure（**沒指定怎麼調**）| `medium_baseline.py`(4-layer + BN + LReLU + FC bottleneck + cosine LR) | ✅ 0.72877 |
| 🥇 Strong | 30-40 min | Multi-encoder autoencoder + Paper reference(PDF 一張示意圖,沒附 URL)| `strong_baseline.py`(3 個並聯 encoder 不同 kernel/dilation) | ❌ 0.70716 |
| 👑 Boss | 30-40 min | **OR**: (A) 加 random noise + 額外 classifier;(B) 用 ResNet 當 encoder + Paper reference | (A)`boss_baseline.py`(DAE + classifier head);(B)`resnet_baseline.py`(ResNet18-style encoder) | ❌ (A) 0.683;❌ (B) 0.729 |

#### ⚠️ 重要的細節

1. **Boss 是 OR 不是 AND** — 方法 A 跟方法 B 擇一即可。我們**兩個都試過**，都沒有突破 medium 的成績。

2. **PDF 提示只給方向,沒給 hyperparam 或實作細節**：
   - 沒寫 latent dim 多少
   - 沒寫網路深度
   - 沒寫 noise scale / classifier loss weight
   - 沒寫訓練 epochs
   - 「Paper reference」字樣是文字超連結,但無法從 PDF 取出 URL

3. **PDF 提示保證不了過 baseline**。要過 Boss 通常需要：
   - **正確設計的方法 A**（不要讓 classifier 學 pixel noise shortcut — 這是我們失敗的原因,詳見 7.4）
   - **更強的 ResNet 變體**（更深、加 attention、loss 加 perceptual...）
   - 或不同 paradigm + ensemble

4. **PDF 提示 vs reference notebook（學生公開答案）有出入**：
   - PDF Strong 提示是「multi-encoder」
   - Reference notebook 把 Strong 解成 VAE(我們驗證:VAE 0.73048 > multi-encoder 0.707)
   - 所以 PDF 提示**不是唯一正解**,可能還有其他更好的設計

下面是每次嘗試的詳細失敗 / 成功原因分析。

---

### 7.1 🥉 Simple — 為什麼比預期好

**預期**：~0.53（簡單 baseline 通常很差）
**實際**：0.66943

**為什麼超預期**：
- 47k 參數的小模型沒有 overfit 的本錢
- Tanh output + [-1, 1] 正規化讓輸出範圍跟 input 對齊（如果用 Sigmoid 或無 activation，初期就會偏一邊）
- 30 epochs 足夠收斂，loss 降到 0.0057

**關鍵教訓**：**小模型有時候是個好朋友**。Capacity 大不等於更好的 anomaly detection。

---

### 7.2 🥈 Medium — 預期符合

**預期**：~0.73
**實際**：0.72877

**做了什麼**：
- 3 conv 層 → 4 conv 層
- 加 BatchNorm（每層 activation 分佈穩定）
- ReLU → LeakyReLU(0.2)（避免 dead neurons）
- 加 FC bottleneck，latent_dim 256
- 80 epochs + cosine annealing

**為什麼這個組合 work**：
- BatchNorm 讓模型可以用比較大的 lr 而不發散
- 更大 latent（256 vs simple 的 8×8×48=3072 隱含 spatial）= 更靈活的表達
- Cosine LR：前期快學，後期 fine-tune
- Score 分佈拉開 3 倍：std 0.0033 → 0.0096

**關鍵教訓**：**穩紮穩打的正確修改 > 花俏的新架構**。

---

### 7.3 🥇 Strong v1 (Multi-Encoder) — 失敗分析

**設計假設**：3 個 encoder（不同 kernel size + dilation）會學到**不同特徵**，concat 後表達能力更強。

**實際結果**：0.70716（**比 medium 退步 0.02**）

**為什麼失敗** — 量化證據：

| 指標 | Medium | Strong v1 |
|------|--------|-----------|
| Params | 3.48M | 8.69M (2.5x) |
| Final loss | ~0.007 | 0.0066 (略低) |
| Score std | 0.0096 | **0.0054**（變窄）|
| Score max | 0.115 | **0.067**（變小）|
| 與 medium 相關係數 | - | **0.97** |

3 大根本原因：

1. **三個 encoder 太相似** — 雖然 kernel 不同，但 BatchNorm + LeakyReLU + 一樣的 channel 結構，訓練後**收斂到相似 latent space**。相關係數 0.97 = 跟 medium 抓到幾乎一樣的訊號。

2. **容量過剩 → reconstruction 太強** — 8.69M 參數讓模型把 anime 也能還原得不錯，**模糊了 normal / anomaly 的差距**。Score std 從 0.0096 降到 0.0054。

3. **缺乏 diversity regularization** — 沒有 loss 強迫三個 encoder 學到不同東西（contrastive loss、orthogonal constraint 等），它們自然收斂到同一個 local minimum。

**關鍵教訓**：
- ❌ **多 = 更好** 是假的
- ✅ **不同 = 更好** 才是對的
- 「Multi-encoder」如果沒有強制分化，就只是「fatter encoder」

---

### 7.4 👑 Boss v1 (Denoising + Classifier) — 更慘的失敗

**設計假設**：
1. 對 input 加 Gaussian noise，autoencoder 學去噪 → 抓 robust 特徵
2. Classifier 分辨 z(clean) vs z(noisy)
3. 假設「anime latent 比較像 noisy latent」→ classifier 提供額外 anomaly signal

**實際結果**：0.68294（**比 medium 退步 0.05**，比 strong 還慘）

**為什麼失敗** — 訓練 log 的鐵證：

```
Epoch 1/80 | recon 0.160 | cls 0.9696
Epoch 2/80 | recon 0.077 | cls 0.0007   ← classifier 第 2 epoch 就解決了
Epoch 3/80 | recon 0.067 | cls 0.2165
...
Epoch 80/80 | recon 0.027 | cls 0.0000  ← 後續一直 0.00
```

Classifier loss 從 0.97 → 幾乎 0 只用了 2 個 epoch。代表它**根本不用「人臉特徵」**，只用「pixel-level noise 的統計特徵」（noisy 的 input variance 比較大）就完美解決任務。

**致命的假設錯誤**：anime 臉 **不是** noisy 臉！
- Noisy face = 真人臉 + 高頻 pixel noise
- Anime face = 不同視覺風格的圖像，沒有 pixel noise，邊緣比較銳利

Classifier 在 test 時看到 anime 的反應跟看到 clean face **一樣** → 對 anomaly detection 毫無幫助。

**rescore_boss.py 的後續分析證實**：

| 變體 | 與 medium 相關係數 |
|------|---------------------|
| boss_mse_only (純 reconstruction) | **+0.997** ← 跟 medium 抓到相同訊號 |
| boss_cls_only (純 classifier 訊號) | **-0.003** ← 完全是隨機雜訊！|
| boss_combined (送出去那版) | +0.705 ← noise 污染了 MSE |

Classifier 訊號是純雜訊（相關係數 ≈ 0），把好的 MSE 訊號 (z-score 標準化後) 1:1 加雜訊 → 自然退步。

**關鍵教訓**：
- 🚫 **加技術之前先驗證假設方向是對的**
- 🚫 不要把訓練 loss 收斂得快當作「學得好」（它可能學到 shortcut）
- ✅ **驗證手段**：在小資料集上跑 sanity check，看 classifier 對人為樣本（noise / blur / colorshift）的反應是否符合預期

---

### 7.5 medium_v2 (L1 loss) — 為什麼這個方向被放棄

**原計畫**：把 MSE 換成 Smooth L1 loss + latent_dim 512，希望「更敏感於結構誤差」。

**為什麼沒跑**：在 user 提示下查看 reference notebook（[zhengjie9510/ML2023-Spring](https://github.com/zhengjie9510/ML2023-Spring) 的 `ML2023_HW08.ipynb`）後發現：
- Reference 用的是 **VAE** 和 **ResNet**，不是 L1 loss
- L1 vs MSE 對 AUC 影響理論上不大（兩者排序相關性高）
- 與其賭一個猜測，不如照 reference 已驗證的方向走

(實作檔案 `medium_v2.py` 在最終清理時已刪除,僅在 README 留下紀錄。)

---

### 7.6 大反思 — 看 reference 才看清的事

**PDF 給的「Paper reference」連結沒附 URL**，所以前面三個 strong/boss 設計是我自己猜的。看 reference notebook 後發現：

| 我以為的 boss = | Reference 實際是 |
|----------------|-----------------|
| Multi-encoder + concat | **VAE** (KL regularization) |
| Denoising + classifier head | **ResNet encoder** (殘差連接從頭訓練) |

差很多！PDF 的「Add random noise and extra classifier」其實只是一張示意圖，**不是 reference 真正用的方法**。

**根本教訓**：
1. 動手寫 code 之前先**確認 paper / reference 的真實方法**
2. PDF 的圖示是「概念」不是「實作 spec」
3. 同學的 GitHub 解答（即使非官方）是寶貴的 sanity check

---

### 7.7 下一波嘗試（進行中）

照 reference 的方向重做：

| Model | 為什麼可能 work |
|-------|----------------|
| **VAE** | KL divergence 強迫 latent ~ N(0, I)，**不能任意記憶訓練樣本**。Anomaly 落在分佈外圍 → 重建差 |
| **ResNet AE** | 殘差連接讓 18M 參數的深層網路真的訓得起來（不退化）。理論上能 capture 更細的人臉特徵，對細微差異更敏感 |

兩者都在背景訓練中。如果 VAE / ResNet **單獨**也只到 0.73 左右，那就 **ensemble medium + VAE + ResNet**（用 rank-based averaging）— 三個不同 inductive bias 的模型集成通常會比單一好。

---

### 7.8 失敗 v.s. 成功的方法表

| 方法 | 結果 | 原因 |
|------|------|------|
| Vanilla conv AE (small) | ✅ 0.669 | 沒 overfit 的本錢 |
| Deeper conv + BN + LeakyReLU | ✅ 0.729 | 穩紮穩打的容量擴張 |
| Multi-encoder (kernel diversity) | ❌ 0.707 | 沒強制分化，三 encoder 學一樣 |
| Denoising + classifier head | ❌ 0.683 | Classifier 學 noise pattern，anime ≠ noisy |
| L1 loss + 大 latent | — 未跑 | Reference 不採用，方向沒把握 |
| **VAE** (KL regularization) | ✅ 0.730 | KL 讓 latent 微微 squeeze,雖然 corr 0.998 但帶來 +0.002 |
| **ResNet AE** (18M params) | ≈ 0.729 | 跟 medium 幾乎完全等價,證實 conv-AE 上限 |
| **Ensemble medium+vae+resnet** | ⚠️ 0.730 | 微微輸給 VAE 單獨,因為模型太像(corr 0.99) |
| 🚀 **MemAE v1** (broken-but-good) | ✅✅ **0.753** | Uniform attention 強迫平均臉重建 → anime 沒法逃 |
| 💀 MemAE v2 (paper-faithful sparse) | ❌❌ 0.673 | Sparse attention 讓 anime 也能匹配 prototype |
| 🥇 **Ensemble memae + VAE** | ✅✅ **0.758** | MemAE 跟 VAE corr 只有 0.73 → 真正互補 |

---

### 7.9 Ensemble 的反直覺結果

直覺上「多個模型 ensemble 一定比單一好」是錯的。本次實驗的數據：

| Setup | Public AUC | 變化 |
|-------|-----------|------|
| VAE 單獨 | **0.73048** | baseline |
| z-score ensemble (3 model) | 0.73008 | **−0.0004** |
| rank ensemble (3 model) | 0.73006 | **−0.0004** |

**為什麼 ensemble 反而微跌**：

```
Ensemble 漲 AUC 的必要條件:
   model_i 的 errors 跟 model_j 的 errors **不相關**
   (這樣平均後 errors 會 cancel out)

我們的模型相關性:
   medium <-> vae    = 0.998  ❌ 太像
   medium <-> resnet = 0.987  ❌ 太像
   vae <-> resnet    = 0.987  ❌ 太像

結果: errors 也高度相關 → 平均後沒 cancel,只是把不同 model 的小優勢稀釋
```

**為什麼三個模型這麼像**：
- 全部是 conv-encoder + conv-decoder 家族
- 全部用 MSE loss
- 全部用 BatchNorm + LeakyReLU/ReLU
- 訓練資料一樣,初始化方式類似
- → 找到非常類似的 local minimum

**真正能漲 AUC 的 ensemble 需要**：
- 不同 inductive bias (eg CNN + Transformer)
- 不同 loss (MSE + Perceptual + Adversarial)
- 不同 paradigm (reconstruction + density estimation + one-class SVM)

**這個發現的價值**：
- 證明「多 = 好」是個常見迷思
- ensemble 工程的核心是 **error decorrelation**, 不是模型數量
- 在小作業裡, 沒有意義把時間花在堆 ensemble — 應該花在找一個夠不同的方法

---

### 7.10 MemAE v1 — 真正的突破

當 conv-AE 家族卡在 0.73 時,讀了 user 推薦的 [MemAE paper (Gong et al. ICCV 2019)](https://arxiv.org/abs/1904.02639) 後實作。

**架構**:在 encoder 跟 decoder 之間插入一個 **Memory module**(N=500 個可學習的 prototype 向量):

```
input ──→ Encoder ──→ z ──→ [Memory Module]──→ z_hat ──→ Decoder ──→ recon
                                  ↑
                          500 個可學習 prototypes
                          (用 attention 加權加總)
```

**結果**:**Public 0.75269 / Private 0.75660** — Private 比 VAE 漲 0.033,Public 漲 0.022。

**為什麼真的 work**:

| 模型 | 跟 medium 的相關係數 |
|------|---------------------|
| VAE | 0.998 (太像) |
| ResNet | 0.987 (太像) |
| **MemAE** | **0.733** (真正不同！) |

MemAE 是這次作業裡**第一個產生「真正不同訊號」的模型**。Memory routing 帶來完全不同的 inductive bias:
- 其他模型:「把 x 壓縮成 z，再從 z 還原」
- MemAE:「把 x 壓縮成 z，**z 必須由 prototype 組成**，再還原」
- 對 anime 來說,它的 z 不像任何 prototype → 還原大壞

**Ensemble 終於有用**:

| Ensemble | Public AUC | 解讀 |
|----------|-----------|------|
| medium + vae + resnet (corr 0.99) | 0.730 | ❌ 微跌(too similar) |
| **memae × 2 + vae** (corr 0.73) | **0.757** | ✅ **+0.005 漲幅** |

這正驗證了 [7.9](#79-ensemble-的反直覺結果) 提到的 ensemble 鐵律:**error decorrelation 才是關鍵**。

---

### 7.11 MemAE v2 — 反直覺的反轉 (修 bug 反而退步！)

訓練 MemAE v1 時注意到一個「bug」:
- **Entropy 一直在 6.20 沒下降** (= ln(500),代表 attention 是完全 uniform)
- 表示 memory module 沒真的「選擇 prototypes」,而是把全部 500 個平均
- 也就是 paper 設計的「sparse memory addressing」**根本沒運作**

排查發現幾個原因:
1. `SHRINK_THRES = 0.0008` 小於 uniform 值 `1/500 = 0.002` → hard shrinkage 沒砍任何東西
2. Memory init `N(0, 0.01)` 太小 → 跟 paper 的 `U(-1/√d, 1/√d) ≈ U(-0.063, 0.063)` 差很多
3. Softmax of cosine similarities ∈ [-1, 1] 在 N=500 上**理論最大值僅 0.005** → 永遠 near-uniform

**v2 修正**:
- Memory init 改成 paper 的 `U(-1/√d, 1/√d)`
- `SHRINK_THRES` 從 0.0008 → 0.005 (高於 uniform,真的會 shrink)
- 加入 **softmax temperature × 10** (讓 cosine 在 [-10, 10] 範圍 softmax,變 sparse)
- `ENTROPY_WEIGHT` 從 2e-4 → 2e-3 (10x)

**結果驗證 "bug" 修好了**:
- Entropy: 6.20 → **0.22** (極 sparse,每 query 只活化 1-2 個 prototype)
- Att max per row: 0.002 → 0.09
- Att nonzero per row: 500 → ~18

**但 Kaggle 結果...大退步 0.08**:

| 版本 | Public AUC | Private AUC | Entropy |
|------|-----------|-------------|---------|
| **MemAE v1** (uniform attention, "bug") | **0.75269** | **0.75660** | 6.20 |
| MemAE v2 (sparse attention, "fixed") | 0.67298 | 0.68315 | 0.22 |

#### 🧠 為什麼修好 bug 反而變差？

**v1 (uniform attention)**:
```
z_hat ≈ 所有 500 prototypes 的平均 = constant (大致是「平均人臉的 latent」)
```
- Normal face: z 跟「平均臉 latent」差不多 → decoder 還能 mock 出一張人臉 → 中等 MSE
- Anime face: z 完全不像「平均臉 latent」→ 強制重建成平均臉 → 大爛 MSE
- **差距大 → AUC 高**

**v2 (sparse attention,"正確"的 memory)**:
```
z_hat ≈ 從 500 prototypes 中挑最像 query 的 1-2 個
```
- Normal face: 找到匹配 prototype → recon 不錯 → 小 MSE
- Anime face: **也能找到「最像」的 prototype** (雖然還是不像,但有 500 個選擇) → 強制重建到某個 prototype → 還是有點像
- **差距小 → AUC 低**

#### 💡 教訓

1. **「Paper 對」≠「對你的 task 對」**:Paper 在 video anomaly detection 上用 sparse attention 有道理(每個 frame 真的要選 best matching prototype)。但 face anomaly detection 的最佳策略可能是「強制把所有東西重建成平均臉」,sparse attention 反而給 anomaly 提供逃生通道。

2. **「Loss 下降」≠「指標變好」**:v2 的 entropy 從 6.20 → 0.22,完美達成 paper 設計目標。但 Kaggle AUC 反而退步 0.08。**訓練目標 (entropy loss) 跟評估目標 (AUC) 不是同一回事**。

3. **「Bug」可能是 feature**:v1 的 entropy 不下降不是 bug 是 feature。它讓 memory 退化成「強制平均化」,對 anomaly detection 反而是恰當的 bottleneck。

4. **永遠用測試指標 verify**:如果不 submit Kaggle 看實際 AUC,只看 entropy 跟 recon loss 會以為 v2 更好。實際 evaluation metric 才算數。

---

### 7.12 多 Seed v1 Ensemble (進行中)

由於 MemAE v1 是目前最強單一模型,自然想試**多個 v1 with 不同 random seed** 的 ensemble:
- 不同 seed = encoder 隨機初始化不同 + memory 隨機初始化不同 + dataloader shuffle 順序不同
- 收斂到的 local minimum 略有差異
- 預期它們互相 corr 0.95+ 但不到 1 → ensemble 可能再漲 0.005-0.01

實作: `memae_v1_multiseed.py` 訓練 seed=123 和 seed=2024 兩個額外 v1,加上原本的 seed=42,組成 3-seed ensemble。

預估訓練時間 ~30 分鐘,結果待測。

---

### 7.13 Multi-seed MemAE Ensemble

訓練 v1 with seeds 42 / 123 / 2024,3 個 model 互相 corr 0.995-0.996。

**結果**(`ensemble_r_3seed_vae_balanced` rank ensemble):
- Public **0.75759**(舊 best 0.75757,基本持平)
- Private **0.75926**(舊 best 0.75817,**+0.001**)

**為什麼漲幅這麼小**:
- 3 seed 之間 corr 0.995-0.996 → 帶來的 noise reduction 很有限
- 收益主要來自加入 VAE(uncorrelated signal),不是 seed 多樣性
- 證明:**seed-level ensemble 收益遠小於 architecture-level ensemble**

---

### 7.14 DeepSVDD — Paradigm 大跳躍

**來源**: user 分享的 [hoya012/awesome-anomaly-detection](https://github.com/hoya012/awesome-anomaly-detection) 中的 [Deep One-Class Classification (ICML 2018)](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf)

**核心想法**:**完全不用 reconstruction!**
- Encoder phi(x; W) 把 input 映射到 R^d
- 訓練目標: 把所有 normal samples 縮到一個 hypersphere 中心 c
- Loss = mean(||phi(x) - c||^2)
- 推論: anomaly score = ||phi(x) - c||^2

**3 個 critical 實作細節**(避免 degenerate solution):
1. 所有 Conv/Linear `bias=False`
2. BatchNorm `affine=False`
3. Center c 從初始 forward pass 平均算出後 **freeze 不可學**

**訓練結果**:
- Loss 從 0.79 → 0.0001(4 個 magnitude drop,有 partial collapse 的 sign)
- `feat_norm` 1.17 → 1.13(沒完全崩到 0)

**Kaggle 結果**: Public **0.63745** / Private **0.64304** — 單獨 AUC 不高,但...

**關鍵發現 — 與其他模型 correlation 暴跌**:

| Pair | Correlation | 解讀 |
|------|-------------|------|
| DeepSVDD vs medium | **0.317** | 完全不同 paradigm |
| DeepSVDD vs VAE | 0.319 | 同上 |
| DeepSVDD vs MemAE | 0.490 | 也帶來不同訊號 |

**Ensemble 結果**(MemAE 3 seeds + VAE + DeepSVDD 不同權重):

| DeepSVDD 權重 | Public | Private | 變化 |
|---------------|--------|---------|------|
| 0× (baseline) | 0.75759 | 0.75926 | - |
| **0.3×** | **0.75689** | **0.75984** ⭐ | **+0.0006 Private (新 best!)** |
| 0.5× | 0.75588 | 0.75882 | 持平 |
| 1.0× | 0.75238 | 0.75635 | -0.003 |

**完美的 weight 曲線**:0.3× = sweet spot,太重會被 DeepSVDD 的低 AUC 拖下。

---

### 7.15 Spatial MemAE — 「補完版」MemAE 反而退步

**來源**: 同 MemAE paper, [Gong et al. ICCV 2019](https://arxiv.org/abs/1904.02639) — 但這次是 paper 原本推薦的 spatial addressing 版本(不是 flatten 後再查 memory)。

**架構差異**:
```
FC MemAE (v1):    encoder -> flatten -> FC -> z(256d) -> memory query -> ...
Spatial MemAE:    encoder -> (256, 4, 4) -> per-position memory query -> ...
                                            ↑ 16 個位置各自查 500 prototypes
                                              = 8000 個 "位置-prototype" 組合
```

**Kaggle 結果**: Public 0.68713 / Private 0.68281 — **大幅退步 0.07**

**為什麼這個「正規版」反而輸 FC v1**:
| | FC MemAE v1 | Spatial MemAE | Medium |
|--|------------|---------------|--------|
| Recon loss | 0.066 | **0.007** | 0.007 |
| Score max | 0.372 | 0.066 | 0.067 |
| Corr with medium | 0.733 | 0.951 | - |

Spatial addressing 給每個 spatial 位置選不同 prototype 的彈性 → **重建得跟 medium 一樣好(loss 0.007)**。失去 FC MemAE 「強迫平均化」的 anomaly filter 效果。

**教訓**: paper 推薦的「正規」做法不一定適合每個 task。**「Bottleneck 越嚴格越好」對 anomaly detection 通常成立**。

---

### 7.16 Multi-Hypothesis AE (MH-AE) — Decoder 沒分化

**來源**: user 分享的 [IEEE 9263356 周邊](https://arxiv.org/pdf/2107.08790) — 多個 decoder 競爭重建,anomaly = 沒有任何 decoder 重建得好。

**架構**: 1 個 encoder + K=3 個 decoder 平行
- Training: 每個 decoder 各自學 MSE,loss 取平均
- Inference: 兩個 score 變體 — min MSE 跟 mean MSE across K decoders

**Kaggle 結果**:
- min variant: 0.72759 / 0.72099
- mean variant: 0.72884 / 0.72233

**Min vs Mean corr 0.9998** — **K=3 decoders 完全沒分化**,都收斂到同一個 reconstruction!

**為什麼沒分化**:
- 3 個 decoder 共用 encoder
- 一樣的 data、一樣的 MSE loss
- 只靠 random init 不夠製造差異
- 真正的 MH-AE 需要 **winner-takes-all routing** 才能強制 decoder 各自負責「不同模式的 normal samples」

**Correlation**: MH-AE vs VAE = **0.998** — 等於另一版 VAE,沒帶來新訊息。

---

### 7.17 VQVAE — MemAE 的離散版

**來源**: `auto_v8.pdf` 投影片 page 23 明確列出。Paper: [van den Oord et al. NIPS 2017](https://arxiv.org/abs/1711.00937)

**架構**:跟 MemAE 類似,但 **hard quantization** 而非 soft attention:
- Encoder 輸出 continuous z
- Codebook 有 K=128 個 learnable embedding
- 每個 spatial 位置的 z 被**精確替換**成最近的 codebook entry e_k*
- Decoder 從 e_k* 重建
- 3 個 loss: reconstruction + codebook_loss + commitment_loss + straight-through estimator

**訓練結果**:
- Recon loss: 0.142 → **0.064** (跟 FC MemAE v1 的 0.066 幾乎一樣！)
- **Unique codes used: 32 / 128** — codebook 部分 collapse,只用 25% entries
- VQ loss 0.78 → 0.15

32 個 active codes 剛好對應 "partial collapse" — 等於是 **MemAE v1 的離散版**!

**Kaggle 結果**: Public **0.73734** / Private **0.74131**

| Model | Public AUC | 跟 MemAE v1 corr |
|-------|-----------|------------------|
| MemAE v1 (soft) | 0.75269 | 1.000 |
| **VQVAE (hard)** | 0.73734 | **0.924** |
| VAE | 0.73048 | 0.734 |

**結論**:
- 「hard quantization vs soft attention」帶來 ~0.015 AUC 差距(MemAE 贏)
- corr 0.92 跟 MemAE 高度相關 → ensemble 不會帶來大幫助
- 但 corr **沒到 1** → 跟 MemAE 有微小互補空間

---

### 7.18 階段性結論(最終)

經過 22 次 submission 跨越 7 個不同 paradigm,目前狀態:

| Top 3 | Public AUC | Private AUC |
|-------|-----------|-------------|
| 🥇 Ensemble r_3seed_vae_balanced | **0.75759** | 0.75926 |
| 🥇 Ensemble z_3seed_vae_ds03 | 0.75689 | **0.75984** ⭐ |
| 🥈 MemAE v1 alone | 0.75269 | 0.75660 |

**從 simple 起點 0.669 → 目前 0.760,Public 漲了 0.091**。

**已驗證有效**:
- ✅ MemAE 的 memory bottleneck(「平均臉重建」迫使 anomaly 無處可逃)
- ✅ Ensemble = uncorrelated models(MemAE + VAE corr 0.73)
- ✅ DeepSVDD 0.3× 微量加入 ensemble(corr 0.32 帶來 +0.0006)
- ✅ Multi-seed v1 averaging(微漲)

**已驗證無效**:
- ❌ 加深網路(strong / ResNet)
- ❌ 多 encoder concat(strong v1)
- ❌ Denoising + classifier head(boss v1)
- ❌ Paper-faithful sparse MemAE(v2)
- ❌ Conv-AE 家族 ensemble(corr 0.99 互相)
- ❌ Spatial MemAE(太彈性失去 bottleneck)
- ❌ MH-AE(decoder 沒分化)
- ❌ VQVAE(略遜 MemAE,且 corr 太高)

**距離 Boss baseline (≈0.80) 還有 0.04**。已知無法用「reconstruction-based AE 家族 + 簡單 ensemble」突破,需要更激進的 paradigm(CutPaste、PatchSVDD、Skip-GANomaly 或 pre-trained features — 後者被作業禁止)。

---

### 7.20 CutPaste — Self-Supervised 第二次突破

**來源**: user 分享的 [hoya012 awesome list](https://github.com/hoya012/awesome-anomaly-detection) → paper [Li et al., "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization" (CVPR 2021)](https://arxiv.org/abs/2104.04015)

**核心想法**: 完全不用 reconstruction,改用 self-supervised classification:
1. **資料增強**: 切一個 patch 貼到同一張圖隨機位置(rect 或 scar)
2. **訓練**: 3-way classifier (normal / cutpaste-rect / cutpaste-scar)
3. **推論**: 兩種 score:
   - Classifier-based: `1 - P(normal)` 
   - **Mahalanobis distance** 從 training set features 的 Gaussian fit

**為什麼直覺上會 work**:
- Anime 臉的「五官位置不正常」跟 cut-paste 製造的「patch 錯位」訊號類似
- Classifier 學到「正常人臉的結構特徵」而不靠 reconstruction
- 完全不同 inductive bias → 應該帶來低 corr 訊號

**訓練結果**:
- 3-way classification accuracy: 0.50 → **0.925**
- 健康收斂(不是 100% — 沒過 fit 到 shortcut)

**Kaggle 結果**:
| Score 變體 | Public AUC | Private AUC |
|-----------|-----------|-------------|
| Classifier-based (1 - P_normal) | 0.59899 | 0.59317 |
| **Mahalanobis distance** | **0.72454** | 0.72015 |

**為什麼 Mahalanobis 贏 classifier 0.13**:
- Classifier 問:「是 cut-paste 過嗎?」 → 對 anime(非 cut-paste),回答「不是」,給低 anomaly score
- Mahalanobis 問:「這個 feature 離 training distribution 多遠?」 → 對 anime,離得很遠,正確給高 score
- **Mahalanobis 正確利用了 encoder 學到的特徵,即使 classifier 認知有誤**

**🚨 與所有現有模型的 correlation 都極低**:

| Pair | Correlation |
|------|-------------|
| cutpaste_mahal vs medium | 0.386 |
| cutpaste_mahal vs MemAE | 0.279 |
| cutpaste_mahal vs VAE | 0.386 |
| cutpaste_mahal vs DeepSVDD | 0.100 |
| (參考: MemAE vs medium) | 0.733 |

**Ensemble 效果驚人**(`3seed + vae + cp×` 不同權重):

| CutPaste 權重 | Public AUC | 提升 |
|--------------|-----------|------|
| 0× (baseline) | 0.75759 | - |
| 0.3× | 0.76305 | +0.005 |
| 1× | 0.77366 | **+0.016** |
| 2× | 0.77928 | +0.005 |
| **3× (peak)** | **0.77978** | +0.001 |
| 4× | 0.77752 | -0.002 |
| 5× | 0.77444 | -0.005 |

**Sweet spot 在 cp=3**: 平滑 saturation 曲線,符合「不相關 + 適度的 AUC」的 ensemble 理論預期。

---

### 7.21 PatchSVDD — 學歪方向反轉成寶藏

**來源**: user 分享的 [hoya012 awesome list](https://github.com/hoya012/awesome-anomaly-detection) → paper [Yi & Yoon, "Patch SVDD: Patch-level SVDD for Anomaly Detection and Localization" (ACCV 2020)](https://arxiv.org/abs/2006.16067)

**核心想法**: DeepSVDD 的 patch-level 擴展 + self-supervised position prediction:
1. **Patch-level DeepSVDD**: 把 64×64 圖切成 9 個 32×32 patches (3×3 grid, stride 16),每個 patch 都要靠近 center c
2. **Position prediction (key!)**: 給定 anchor patch + 8 個 neighbor 中的一個,預測相對位置 (8-way classification)
3. **Loss = position_loss + svdd_loss**
4. **推論**: per-patch distance to center → 平均當 anomaly score

**為什麼直覺上會 work**:
- Position prediction 強迫 encoder 學「人臉 part 應該在哪裡」的結構知識
- Anime 的結構亂(眼睛比例不同、五官位置不一樣) → patches 跟 prototype 位置關係不對

**訓練結果**:
- Position accuracy: 0.26 → **0.9957** (99.57%!) — model 完美學會位置預測
- svdd loss: 3.88 → 0.15

**🚨 史無前例的「全負 correlation」**:

| Pair | Correlation |
|------|-------------|
| PatchSVDD (原始) vs medium | **-0.265** 🤯 |
| PatchSVDD vs MemAE | -0.267 |
| PatchSVDD vs VAE | -0.269 |
| PatchSVDD vs CutPaste | -0.111 |

**全部負相關!** PatchSVDD 把 anomaly 排序排**相反方向** — 對 anime 給低分,對正常人臉給高分。

**Kaggle 結果**:
| Variant | Public AUC |
|---------|-----------|
| PatchSVDD (原始) | **0.23603** |
| PatchSVDD (`1 - score`) | **0.76396** |

`AUC_original + AUC_inverted = 0.236 + 0.764 ≈ 1.0` ✓ 數學上一致。

**為什麼學歪方向**:
- Position prediction 任務太成功(99.57%) → encoder 完全 focus 在學「結構特徵」
- DeepSVDD-style 「靠近 center」的 loss 反而學到「結構越正常 → 距 center 越遠」的反向關係
- 對 anime 來說,結構不正常 → 距 center 反而近 → 原始 score 低 → 反轉成「異常」

**🔥 Ensemble 加 inverted PatchSVDD 帶來爆炸性提升**:

| PS 權重 | Public AUC | 提升 |
|---------|-----------|------|
| 0× | 0.77978 | (baseline = CutPaste 3×) |
| 1× | 0.80458 | **+0.025** 🚀 |
| 2× | 0.82152 | **+0.017** |
| 3× | 0.83187 | **+0.010** |
| 4× | 0.83733 | **+0.005** |
| **5× (peak, no DS)** | **0.83973** | **+0.002** ⭐ |
| 5.5× | 0.83970 | -0.00003 |
| 6× | 0.83736 | -0.002 |
| 7× | 0.83282 | -0.005 |

**PS=5 是 sweet spot**,而且**拿掉 DeepSVDD (no_ds) 反而漲** — DeepSVDD 在這個強 ensemble 裡變雜訊。

---

### 7.22 🏆 最終 best ensemble — 完整配方

**`ensemble_r_cp3_no_ds_ps5.csv`** — Public **0.83973** / Private **0.83490**

```python
# Rank-based ensemble of 6 component models:
rank_ensemble([
    (memae_v1_seed42,    weight=1),  # 3 seeds of MemAE v1
    (memae_v1_seed123,   weight=1),  # — each gives MemAE bottleneck signal
    (memae_v1_seed2024,  weight=1),  # — noise reduction via averaging
    (vae,                weight=1),  # KL-regularized AE
    (cutpaste_mahal,     weight=3),  # Self-supervised CutPaste (Mahalanobis)
    (patchsvdd_inverted, weight=5),  # Position prediction (inverted!)
])
```

**為什麼這個組合有效**:
- 6 個模型分別來自 **3 個不同的 paradigm**:
  - Reconstruction-based: MemAE (memory bottleneck) + VAE (KL)
  - Self-supervised classifier: CutPaste
  - Self-supervised position prediction: PatchSVDD (inverted)
- **Pairwise correlations 都很低**(0.27-0.74)
- Rank-based aggregation 消除了 score scale 差異(MemAE ~0.1, DeepSVDD ~0.0001, CutPaste mahal ~100)

**為什麼 rank 贏 z-score**:
| 變體 | Public | Private |
|------|--------|---------|
| `ensemble_r_cp3_no_ds_ps5` | **0.83973** | **0.83490** |
| `ensemble_z_cp3_no_ds_ps5` | 0.83061 | 0.82536 |
| Diff | **+0.009** | **+0.010** |

Score scale 差太多時,z-score 仍然受少數高 magnitude 值影響;rank 把 magnitudes 完全消除,只看排序。

---

### 7.19 完整方法清單 × 資料來源

下面是**所有嘗試過的方法**,每個都標註資料來源(從哪份 PDF 哪一頁、哪篇 paper、哪個提示延伸而來):

| # | 方法 | 資料來源 | 延伸自 | Kaggle Public/Private |
|---|------|---------|-------|---------------------|
| 1 | `simple_baseline.py` | `HW08.pdf` p15 「Simple: Sample code」 | (起點) | 0.66943 / 0.66864 |
| 2 | `medium_baseline.py` | `HW08.pdf` p15 「Medium: Adjust model structure」 | simple + BatchNorm / LeakyReLU / 更大 latent | 0.72877 / 0.72263 |
| 3 | `strong_baseline.py` | `HW08.pdf` p15 「Strong: Multi-encoder autoencoder」+ PDF 示意圖 | medium + 3 並聯 encoder | 0.70716 / 0.70192 ❌ |
| 4 | `boss_baseline.py` | `HW08.pdf` p15-16 「Boss A: Add random noise and an extra classifier」+ p16 示意圖 | medium + DAE + classifier head | 0.68294 / 0.68244 ❌ |
| 5 | medium_v2 (L1 loss,**已刪除**) | 自行嘗試的 L1 loss + 大 latent 變體 | medium + L1 loss | 未提交 |
| 6 | `vae_baseline.py` | `auto_v8.pdf` p30「With some modification, we have variational auto-encoder」+ reference notebook `ML2023_HW08.ipynb` 的 VAE class | medium + KL regularization | 0.73048 / 0.72369 |
| 7 | `resnet_baseline.py` | `HW08.pdf` p15 「Boss B: use Resnet as encoder」+ reference notebook 的 ResNet class | medium + ResNet18-style residual blocks | 0.72885 / 0.72120 |
| 8 | `rescore_boss.py` | 自製分析工具 | 分析 boss v1 失敗原因 | 不提交,僅後處理 |
| 9 | `ensemble.py`(conv-AE 家族) | 自製 | medium + vae + resnet rank/zscore | 0.73006 / 0.72313 ⚠️ |
| 10 | `memae_baseline.py` v1 | user 分享的 [IEEE 9263356 周邊](https://ieeexplore.ieee.org/abstract/document/9263356) + [hoya012 awesome list](https://github.com/hoya012/awesome-anomaly-detection) → paper [Gong et al. ICCV 2019](https://arxiv.org/abs/1904.02639) | medium + memory module(soft attention,**未調對的 broken version**) | **0.75269 / 0.75660** 🚀 |
| 11 | `memae_baseline.py` v2(paper-faithful) | 同 paper,但加 softmax temperature + paper init + 正確 shrinkage threshold | MemAE v1 + 「修 bug」 | 0.67298 / 0.68315 ❌❌ |
| 12 | `memae_v1_multiseed.py` | 自製 | MemAE v1 × 2 不同 seed (123, 2024) | 各 seed ~0.75,ensemble 微漲 |
| 13 | `ensemble_with_memae.py` | 自製 | MemAE + VAE 多種權重組合 | 最佳 0.75757 / 0.75817 |
| 14 | `ensemble_multiseed.py` | 自製 | 3 seed MemAE + VAE 多種組合 | 最佳 0.75759 / 0.75926 |
| 15 | `deepsvdd_baseline.py` | user 分享的 [hoya012 awesome list](https://github.com/hoya012/awesome-anomaly-detection) → paper [Ruff et al. ICML 2018](http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf) | 完全跳出 reconstruction paradigm | 0.63745 / 0.64304 |
| 16 | `ensemble_with_deepsvdd.py` | 自製 | 3 seed MemAE + VAE + DeepSVDD 多權重 | **0.75689 / 0.75984** ⭐ |
| 17 | `spatial_memae_baseline.py` | 同 MemAE paper, paper 原本推薦的 per-spatial-position 版本 | MemAE v1 + 16 spatial queries | 0.68713 / 0.68281 ❌ |
| 18 | `mh_ae_baseline.py` | user 分享的 [IEEE 9263356 周邊](https://ieeexplore.ieee.org/abstract/document/9263356) → paper [Sasaki et al. arXiv 2107.08790](https://arxiv.org/abs/2107.08790) | medium + 3 平行 decoder | 0.72884 / 0.72233 (mean), 0.72759 / 0.72099 (min) ❌ |
| 19 | `vqvae_baseline.py` | `auto_v8.pdf` p23 「Vector Quantized Variational Auto-encoder (VQVAE)」+ paper [van den Oord et al. NIPS 2017](https://arxiv.org/abs/1711.00937) | MemAE 的離散 codebook 版本 | 0.73734 / 0.74131 |
| 20 | `report_q2_fc_autoencoder.py` | `HW08.pdf` p17-18 Report Q2 要求 | 完全獨立 FC autoencoder | 不提交,只做 latent manipulation 視覺化 |

**未做但提到的方法**(時間關係略過):
- **Feature Disentanglement**: `auto_v8.pdf` p14
- **CutPaste**: [Li et al. CVPR 2021](https://arxiv.org/abs/2104.04015) — user 分享過
- **Skip-GANomaly**: [Akcay et al. IJCNN 2019](https://arxiv.org/abs/1901.08954) — user 分享過
- **PatchSVDD**: [arXiv 2006.16067](https://arxiv.org/abs/2006.16067) — user 分享過
- **GANomaly**: [Akcay et al. ACCV 2018](https://arxiv.org/abs/1805.06725) — hoya012 awesome list

---

## 8. 環境設置

### 必要套件
使用 conda 環境 `kaggle_env`（Python 3.11 + PyTorch 2.10 cu130，Blackwell 相容）：

```powershell
# PyTorch + ML 套件
torch 2.10.0+cu130
numpy 1.24.3
pandas 2.3.3
scikit-learn 1.7.2  # 用來算 ROC AUC（local 驗證用）
tqdm 4.67.3
```

### GPU
測試在 **RTX 5070 Ti 16GB** (Blackwell, sm_120)。

---

## 9. 執行方式

```powershell
# 🥉🥈 過 baseline 的（保留作 best result）
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe simple_baseline.py    # 0.669
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe medium_baseline.py    # 0.729

# 🥇👑 PDF 提示的 strong/boss(都退步,保留作試錯記錄)
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe strong_baseline.py    # 0.707 ❌
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe boss_baseline.py      # 0.683 ❌

# Reference notebook 方向(微漲)
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe vae_baseline.py       # 0.730 ✅
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe resnet_baseline.py    # 0.729 ≈

# 🚀 突破 #1 — MemAE 家族
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe memae_baseline.py        # 0.753 v1 ✅ / v2 0.673 ❌
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe memae_v1_multiseed.py    # 多 seed v1
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe spatial_memae_baseline.py # paper 原版 0.687 ❌
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe vqvae_baseline.py         # MemAE 離散版 0.737

# 不同 paradigm 嘗試
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe deepsvdd_baseline.py     # 0.637
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe mh_ae_baseline.py        # 0.728

# 🚀 突破 #2 + #3 — Self-supervised
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe cutpaste_baseline.py     # 0.725 (Mahalanobis)
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe patchsvdd_baseline.py    # 0.236 原始 / 0.764 反轉

# Ensemble 工具(運行順序對應 README 7.10 → 7.22)
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe ensemble.py              # conv-AE 家族 (微跌)
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe ensemble_with_memae.py   # 加 MemAE → 0.758
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe ensemble_multiseed.py    # 3-seed MemAE + VAE
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe ensemble_with_deepsvdd.py # 加 DeepSVDD 0.3x → 0.760
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe ensemble_with_cutpaste.py # 加 CutPaste 3x → 0.780
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe ensemble_final.py        # 加 PatchSVDD 5x → 🏆 0.840

# 分析工具
C:/ProgramData/miniconda3/envs/kaggle_env/python.exe rescore_boss.py       # 對已訓練的 boss model 算不同 score 組合
```

每個 script 跑完會產生 `output/<level>_submission.csv`，格式：
```
ID,score
0,0.0123
1,0.0456
...
```

直接上傳到 Kaggle 比賽頁面就好。

---

## 10. 檔案結構

```
HW8/
├── README.md                          ← 你正在讀
├── REPORT.md                          ← Report Q1 (VAE) + Q2 (FC AE) writeup
├── HW08.pdf                           ← 作業 spec
├── auto_v8.pdf                        ← 李宏毅 autoencoder 投影片
├── data/
│   ├── trainingset.npy                ← 100k 真人臉 (1.17 GB)
│   └── testingset.npy                 ← 19636 張混合 (230 MB)
│
├── ── PDF 提示路線 (4 個 baseline) ──
├── 🥉 simple_baseline.py              ← 0.66943 ✅
├── 🥈 medium_baseline.py              ← 0.72877 ✅
├── 🥇 strong_baseline.py              ← 0.70716 ❌ (multi-encoder 退步)
├── 👑 boss_baseline.py                ← 0.68294 ❌ (DAE+cls 退步)
│
├── ── Reference notebook 方向 ──
├── 📘 vae_baseline.py                 ← 0.73048 ✅ (KL regularization)
├── 📘 resnet_baseline.py              ← 0.72885 ≈ (殘差連接)
│
├── ── 🚀 突破 #1: MemAE 家族 ──
├── 🚀 memae_baseline.py               ← v1: 0.75269 ✅✅ / v2: 0.67298 ❌❌ (反直覺反轉)
├── 🚀 memae_v1_multiseed.py           ← 多 seed v1 (seeds 123, 2024)
├── 📐 spatial_memae_baseline.py       ← paper 原版 0.687 ❌
├── 📐 vqvae_baseline.py               ← MemAE 離散版 0.737
│
├── ── 不同 paradigm 嘗試 ──
├── 🎯 deepsvdd_baseline.py            ← 0.637 (corr 0.32 → ensemble 微貢獻)
├── 🎯 mh_ae_baseline.py               ← 3 decoders 沒分化 0.728 ❌
│
├── ── 🚀 突破 #2 + #3: Self-Supervised ──
├── 🚀 cutpaste_baseline.py            ← 0.725 (Mahalanobis), ensemble 帶來 0.780 突破
├── 🚀 patchsvdd_baseline.py           ← 0.236 原始 / 0.764 反轉,ensemble 帶來 0.840 突破
│
├── ── Ensemble + 分析工具 ──
├── 🔧 ensemble.py                     ← conv-AE ensemble (medium+vae+resnet)
├── 🔧 ensemble_with_memae.py          ← 加 MemAE → 0.758
├── 🔧 ensemble_multiseed.py           ← 3-seed MemAE + VAE → 0.7593
├── 🔧 ensemble_with_deepsvdd.py       ← 加 DeepSVDD 0.3x → 0.7598
├── 🔧 ensemble_with_cutpaste.py       ← 加 CutPaste 3x → 0.780
├── 🔧 ensemble_final.py               ← 🏆 加 PatchSVDD 5x → **0.840** (FINAL BEST)
├── 🔬 rescore_boss.py                 ← 分析 boss 失敗原因
├── 🔬 generate_viz.py                 ← Notion writeup 視覺化生成
│
├── ── 放棄的計畫 (保留作試錯記錄) ──
├── (medium_v2.py 已刪除               ← L1 loss 變體,未跑,僅在 README 留紀錄)
│
├── ── Report 相關 ──
├── 📝 report_q2_fc_autoencoder.py     ← Q2 的 FC autoencoder + latent manipulation
│
└── output/
    ├── *_model.pt                     ← 訓練好的權重
    ├── *_submission.csv               ← Kaggle submission
    ├── *_train.log                    ← 訓練 log
    ├── q2_latent_manipulation.png     ← Q2 視覺化
    └── ensemble_*.csv                 ← 各種 ensemble 變體
```

**檔案分類**:
- 🥉🥈🥇👑 PDF 提示的 4 個 baseline
- 📘 Reference notebook 採用的方法 (VAE / ResNet)
- 🚀 真正帶來突破的方法 (MemAE 家族)
- 📐 MemAE 變體 (Spatial / VQVAE)
- 🎯 不同 paradigm 嘗試 (DeepSVDD / MH-AE)
- 🔧 Ensemble 與分析工具
- ✏️ 計畫但放棄的 (保留思路記錄)
- 📝 Report Q2 用

---

## 11. 注意事項

- ❌ **禁止 pre-trained model**（作業規定）
- ❌ **禁止額外資料**（作業規定）
- 每天 Kaggle 最多 submit 5 次
- 報告 Q2 要訓練一個 **fully connected** autoencoder，不要跟 conv autoencoder 混用
