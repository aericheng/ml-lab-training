import shutil
import os

# 設定來源與目的路徑
source_dir = "/kaggle/input/datasets/aericheng/my-best-checkpoint"
dest_dir = "/kaggle/working" # 這是程式預設的 Output 區

print(f"🚀 準備從 [{source_dir}] 複製檔案到 [{dest_dir}] ...\n")

# 開始複製
if os.path.exists(source_dir):
    file_count = 0
    for filename in os.listdir(source_dir):
        # 我們只複製 .ckpt 結尾的模型檔
        if filename.endswith(".ckpt"):
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(dest_dir, filename)
            
            # 複製檔案 (如果已經存在會覆蓋，確保是用你上傳的)
            shutil.copy(src_path, dst_path)
            print(f"✅ 成功複製: {filename}")
            file_count += 1
            
    if file_count == 0:
        print("⚠️ 警告：路徑存在，但裡面沒有找到 .ckpt 檔案！")
    else:
        print(f"\n🎉 搬運完成！共複製了 {file_count} 個檔案。")
        print("現在你的 Ensemble 程式碼可以讀到這些模型了！")
        
else:
    print(f"❌ 錯誤：找不到路徑 [{source_dir}]")
    print("👉 請檢查 `dataset_name` 變數是否打對？")
    print("👉 請確認右側 Input 區塊真的有這個資料夾。")

!nvidia-smi
_exp_name = "resnet18_scratch_mixup"
# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models
# This is for the progress bar.
from tqdm.auto import tqdm
import random

from sklearn.model_selection import KFold
import glob
myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet 標準輸入大小
    transforms.ToTensor(),
    # ImageNet 的平均值與標準差
    # ResNet18 模型在ImageNet上訓練時，圖片全部都經過這組數字的標準化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # RandomResizedCrop會隨機裁切並縮放
    # 強迫模型去學物體的局部特徵 (如麵條的紋理)，而不只是看輪廓
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), # 更寬的縮放比例

    # 隨機水平翻轉
    transforms.RandomHorizontalFlip(),

    # 隨機旋轉
    transforms.RandomRotation(20),

    # 隨機調整亮度、對比、飽和度
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 使用自動增強策略，涵蓋了 ColorJitter 且更多樣化
    transforms.TrivialAugmentWide(),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    # 加入隨機擦除，對抗過擬合非常有幫助
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])
class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im,label
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 載入ResNet18
        self.model = models.resnet18(weights=None)

        # ResNet18 的最後一層輸入是512維，輸出原本是1000類
        # 換成11類
        self.model.fc = nn.Linear(512, 11)

    def forward(self, x):
        # ResNet 的 forward 已經寫好了，直接呼叫即可
        return self.model(x)
    
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)

# The number of batch size.
batch_size = 64

# The number of training epochs. initial n_epochs = 8
n_epochs = 200

# If no improvement in 'patience' epochs, early stop.
patience = 15

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
lr = 0.0003
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# 這行會讓 LR 像餘弦波一樣慢慢下降，最後降到 0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

import wandb
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    print("Successfully logged into WandB!")
except:
    print("Run without Secrets: Please input your API key manually.")
    wandb.login()

project_name = "Food-Classification-KFold"

# 1. 取得所有訓練圖片路徑 (合併原本的 train 和 valid 資料夾，因為我們要重新分配)
# 假設你的資料放在 /kaggle/input/ml2023spring-hw3/
all_files = sorted(glob.glob("/kaggle/input/ml2023spring-hw3/train/*.jpg") + 
                   glob.glob("/kaggle/input/ml2023spring-hw3/valid/*.jpg"))
all_labels = [int(f.split("/")[-1].split("_")[0]) for f in all_files] # 解析標籤用於分層

# 2. 設定 K-Fold (例如 5 Fold)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 這會變成一個 list，裡面有 5 組 (train_index, val_index)
folds = list(kfold.split(all_files, all_labels))

print(f"總共找到 {len(all_files)} 張圖片，將進行 {n_splits}-Fold Cross Validation")

target_folds = [0, 1, 2, 3, 4]

for fold, (train_idx, val_idx) in enumerate(folds):

    if fold not in target_folds:
        print(f"Skipping Fold {fold+1} (Already trained)...")
        continue
    
    print(f"\n===== Start Training Fold {fold+1}/{n_splits} =====")

    run = wandb.init(
        project=project_name,
        name=f"fold_{fold+1}_{_exp_name}",
        config={
            "fold": fold + 1,
            "lr": lr,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "architecture": "Classifier"
        },
        reinit=True # 每個 Fold 開啟新的 Run
    )

    # 根據 index 挑出檔案
    train_files = [all_files[i] for i in train_idx]
    val_files   = [all_files[i] for i in val_idx]

    train_set = FoodDataset(path=None, tfm=train_tfm, files=train_files)
    valid_set = FoodDataset(path=None, tfm=test_tfm,  files=val_files) # valid 用 test_tfm

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 初始化一個全新的模型, 重置權重
    model = Classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):
    
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
    
        # These are used to record information in training.
        train_loss = []
        train_accs = []
    
        for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}"):
    
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)
    
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))
    
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))
    
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
    
            # Compute the gradients for parameters.
            loss.backward()
    
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
    
            # Update the parameters with computed gradients.
            optimizer.step()
    
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    
            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
    
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
    
        # scheduler.step()要放在 optimizer.step() 之後
        scheduler.step()
        # Print the information.
        current_lr = scheduler.get_last_lr()[0]
        print(f"[ Fold {fold+1} | {epoch + 1:03d}/{n_epochs:03d} ] Train loss = {train_loss:.5f}, acc = {train_acc:.5f}, lr = {current_lr:.6f}")
    
        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
    
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
    
        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader, desc="Validating"):
    
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
    
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))
    
            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))
    
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    
            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break
    
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "valid/loss": valid_loss,
            "valid/acc": valid_acc,
            "lr": current_lr
        })
    
        # Print the information.
        print(f"[ Fold {fold+1} | {epoch + 1:03d}/{n_epochs:03d} ] Valid loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    
    
        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Fold {fold+1} | {epoch + 1:03d}/{n_epochs:03d} ] Valid loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Fold {fold+1} | {epoch + 1:03d}/{n_epochs:03d} ] Valid loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    
    
        # save models
        if valid_acc > best_acc:
            wandb.run.summary["best_accuracy"] = valid_acc
            print(f"Fold {fold+1} Best model found at epoch {epoch+1}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_fold{fold+1}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break

    run.finish()

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("/kaggle/input/ml2023spring-hw3/test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 隨機增強的測試資料 (用 train_tfm 來考驗模型應變能力) -> 權重 0.2
test_set_aug = FoodDataset("/kaggle/input/ml2023spring-hw3/test", tfm=train_tfm)
test_loader_aug = DataLoader(test_set_aug, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 1. 載入所有 5 個模型
models_list = []
for fold in range(4):
    model_temp = Classifier().to(device)
    # 載入你剛剛練好的權重
    model_temp.load_state_dict(torch.load(f"{_exp_name}_fold{fold+1}_best.ckpt"))
    model_temp.eval()
    models_list.append(model_temp)

print("Start Ensemble Prediction with TTA...")
predictions = []

# 把結果存成 List，最後再一起加總
clean_preds = [] # 存原圖的預測
aug_preds   = [] # 存增強圖的預測 (會累積 5 次)

with torch.no_grad():
    for data, _ in tqdm(test_loader):
        data = data.to(device)
        
        # 用來累積 5 個模型的 logits
        avg_logits = torch.zeros(data.shape[0], 11).to(device)
        
        # 針對每個模型進行預測
        for model in models_list:
            avg_logits += model(data)

        clean_preds.append(avg_logits / len(models_list))
        
    aug_preds = [torch.zeros_like(logits) for logits in clean_preds]
    tta_times = 5

    for t in range(tta_times):
        print(f"   TTA Round {t+1}/{tta_times}")
        for i, (data, _) in enumerate(test_loader_aug):
            data = data.to(device)
            avg_logits = torch.zeros(data.shape[0], 11).to(device)
            
            for model in models_list:
                avg_logits += model(data)
            
            aug_preds[i] += (avg_logits / len(models_list))
    
    for i in range(len(aug_preds)):
        aug_preds[i] /= tta_times

    for i in range(len(clean_preds)):
        # Final = Clean * 0.8 + Aug * 0.2
        final_logits = (clean_preds[i] * 0.8) + (aug_preds[i] * 0.2)
        
        test_label = np.argmax(final_logits.cpu().data.numpy(), axis=1)
        predictions += test_label.squeeze().tolist()

# 存檔
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = predictions
df.to_csv("submission_ensemble_tta.csv", index=False)
