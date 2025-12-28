# 复现基本信息
github:
https://github.com/daniel4725/HyperFusion  
  
Paper:
https://arxiv.org/abs/2403.13319  
  
数据集：
http://brain-development.org/ixi-dataset/  


# 1.环境安装Install
```
conda create -n hyperfusion python=3.10
conda activate hyperfusion
```
```
# 安装 PyTorch + CUDA
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```
```
# 安装 Lightning 与辅助库
pip install pytorch-lightning==2.3.0 torchmetrics==1.3.0 wandb==0.17.0
pip install scikit-learn==1.5.0 easydict==1.10 pyyaml==6.0.2
pip install matplotlib==3.8.4 seaborn==0.13.2 tqdm==4.66.5 pandas==2.2.2
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==2.3.0 torchmetrics==1.3.0 wandb==0.17.0
pip install scikit-learn==1.5.0 easydict==1.10 pyyaml==6.0.2
pip install matplotlib==3.8.4 seaborn==0.13.2 tqdm==4.66.5 pandas==2.2.2
```
# 2.data介绍与上传
IXI的T1MRI，MRI数据，3D，一共18GB,NIFTI格式  
  
训练集，验证集，测试集均从中划分，比例为8:1:1，格式需从nii.gz转成npy


# 3.复现步骤实现
## 3.1 下载数据集
### HBN

```$ datalad clone \
    https://github.com/ReproBrainChart/<study>_<content>.git \
    -b <qc_threshold>-<version>
```
但是下载的nii.gz里只有网址，改了很多次没有成功，ai说windows datalad无法连接fcp-indie要用linux
### SLIM
这个存在亚马逊S3桶用cyber duck要钱在找不要钱的方法
### IXI
http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
## 报错
### 3.1.1
```
问题：polyp标注数大于图片数，但一个图片应该只有一个标注
原因：(ttdg) root@autodl-container-0fc04ca4ba-2313fd84:~/autodl-tmp# 
python /root/autodl-tmp/plpolypdajs.py unique pixel values: 
[ 0 1 2 3 4 5 6 7 8 248 249 250 251 252 253 254 255] connected components (excluding bg): 1 findContours returned: 1
掩码不是纯黑白二值图，而是被保存成了灰度渐变或者有轻微压缩误差的图像（比如 JPEG 保存过、或者 PNG 压缩出了一堆伪像）。

解决方法：强行二值化
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
mask = np.where(mask > 128, 255, 0).astype(np.uint8)
```
### 3.1.2
```
问题:
fundus的标注数量应该是图片数量两倍，但是多很多

原因：把很小的连通区域也标进去了

解决方法：
忽略小碎片
num_labels, labels = cv2.connectedComponents(binm)
area_thresh = binm.size * 0.001  # 小于0.1%像素面积的忽略
counts = 0
for i in range(1, num_labels):
    if np.sum(labels == i) >= area_thresh:
        counts += 1

```
```
问题：
Assertion `t >= 0 && t < n_classes` failed
标签和模型输出类别数不匹配

解决方法：
验证集设置错误，改一下验证集
```
```
问题：
ValueError: Does not validate against any of the Union subtypes
Subtypes: [<class 'NoneType'>, <class 'lightning.pytorch.core.module.LightningModule'>]
Errors:
  - Expected a <class 'NoneType'>
  - 'init_args'
Given value type: <class 'jsonargparse._namespace.Namespace'>
Given value: Namespace(class_path='chest_xray.pl_modules.DomainConditionedCXRLitModule', init_args=Namespace(...))

解决方法：这是用ctrl C中断训练导致权重不完整导致的，完整训练就可以
```
```
问题：
KeyError: np.int64(1)
wandb.log({"conf_mat": confusion_matrix(preds=val_domain_preds, y_true=val_domain_labels, class_names=self.cls_names)})

解决方法：
类别映射那里修改了一下并且如果训练时只有一个域就不画混淆矩阵
```
## 3.2  nii.gz转npy
```
import nibabel as nib
import numpy as np
import os

src_dir = "/root/autodl-tmp/your_raw_mri_folder"   # 你的 T1w.nii.gz 所在处
dest_dir = "/root/autodl-tmp/dataaa/mri"
os.makedirs(dest_dir, exist_ok=True)

for file in os.listdir(src_dir):
    if not file.endswith(".nii.gz"):
        continue

    subject = file.replace(".nii.gz", "")
    print("Converting", subject)

    nii = nib.load(os.path.join(src_dir, file))
    arr = nii.get_fdata().astype(np.float32)

    np.save(os.path.join(dest_dir, f"{subject}.npy"), arr)

```

## 3.3 training
```
python /root/autodl-tmp/HyperFusion-main/train.py -c experiments/brain_age_prediction/default_train_config.ymlconfigs/train/hyper_parameters_stage1.yaml']"
```


## 报错
### 3.3.1
```
config里没有设置strategy，加auto
Traceback (most recent call last):
 File "/root/autodl-tmp/HyperFusion-main/train.py", line 197, in <module> main(config) File "/root/autodl-tmp/HyperFusion-main/train.py", line 56, in main trainer = pl.Trainer( File "/root/miniconda3/envs/hyperfusion/lib/python3.10/site-packages/pytorch_lightning/utilities/argparse.py", line 70, in insert_env_defaults return fn(self, **kwargs) File "/root/miniconda3/envs/hyperfusion/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 400, in __init__ self._accelerator_connector = _AcceleratorConnector( File "/root/miniconda3/envs/hyperfusion/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py", line 130, in __init__ self._check_config_and_set_final_flags( File "/root/miniconda3/envs/hyperfusion/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py", line 193, in _check_config_and_set_final_flags raise ValueError( ValueError: You selected an invalid strategy name: strategy=None. It must be either a string or an instance of pytorch_lightning.strategies.Strategy. Example choices: auto, ddp, ddp_spawn, deepspeed, 
... Find a complete list of options in our documentation at https:
```
### 3.3.2
```
NotImplementedError: Support for `training_epoch_end` has been removed in v2.0.0. 
`PlModelWrapBrainAge` implements this method. 
You can use the `on_train_epoch_end` hook instead.
原因＋方法：
使用的 PlModelWrapBrainAge 里实现了 training_epoch_end(self, outputs) 方法。

从 PyTorch Lightning v2.0.0 开始，training_epoch_end 已经被移除，不再被调用。

Lightning 官方建议使用 on_train_epoch_end hook 代替
def training_epoch_end(self, outputs):
    ...
    改为
def on_train_epoch_end(self):
    # outputs 可以通过 self 在 step() 中累积
    pass
```
### 3.3.3
```
RuntimeError: Trying to resize storage that is not resizable
RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x752640 and 39424x16)
输入尺寸不一致，代码里没有原文的crop操作
target_shape = (90, 120, 99)

# 对 H、W、D 分别 pad 或 crop
img_padded = np.zeros(target_shape, dtype=img.dtype)

# 取 min 区域
h, w, d = img.shape
img_padded[:min(h,target_shape[0]),
           :min(w,target_shape[1]),
           :min(d,target_shape[2])] = img[:target_shape[0],
                                          :target_shape[1],
                                          :target_shape[2]]

img = img_padded
3.3.4
wandb: Run summary: wandb: epoch 69
 wandb: train/MAE 2.55588 wandb: train/loss 11.03229 wandb: trainer/global_step 69 wandb: val/MAE 39.81068 wandb: val/MSE 1584.89062 
wandb: val/best_MAE 37.62508 wandb: val/best_MSE 1415.64697 wandb:
config里面partial_data打开了，注释掉就行
Evaluation
python /root/autodl-tmp/HyperFusion-main/eval.py -c /root/autodl-tmp/HyperFusion-main/experiments/brain_age_prediction/test_config.ymlrun --config_file "['configs/supported_eval/infer_patch_autopoint.yaml']" --dataset_name 'xxxx'
```
## 3.4 Results
指标为MAE 
|       | 主表 | 我的 |
| ----- | ---- | ---- |
| female+male | 2.55 | 5.00 |
| female      | 2.25 | 5.52 |
| male        | 2.4  | 4.29 |
