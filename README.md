# 复现基本信息
github:
https://github.com/Project-MONAI/VISTA  
Paper:
https://www.alphaxiv.org/abs/2406.05285  
数据集：
https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
用的MSD的脾脏数据  

# 1.环境安装Install
```git clone https://github.com/Project-MONAI/VISTA.git
cd ./VISTA/vista3d
conda create -n -y vista3d python=3.9
conda activate vista3d
pip install -r requirements.txt
``` 
# 2.data介绍与上传
MSD的脾脏数据集，CT数据，3D，一共1.5GB  
训练集，验证集，测试集均从中划分，比例为64:16:20，格式为nii.gz


# 3.复现步骤实现
## 3.1 生成超体素

＃step1. Add this function to predictor.py/SamPredictor
```
@torch.no_grad()
def get_feature_upsampled(self, input_image=None):
    if input_image is None:
        image_embeddings = self.model.mask_decoder.predict_masks_noprompt(self.features)
    else:
        image_embeddings = self.model.mask_decoder.predict_masks_noprompt(self.model.image_encoder(input_image))
    return image_embeddings
```
   
＃step2    Add this function to modeling/mask_decoder.py/MaskDecoder
```
def predict_masks_noprompt(
    self,
    image_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predicts masks. See 'forward' for more details."""
    # Concatenate output tokens

    # Expand per-image data in batch direction to be per-mask
    src = image_embeddings
    # Upscale mask embeddings and predict masks using the mask tokens
    upscaled_embedding = self.output_upscaling(src)

    return upscaled_embedding
```    
step3 Run the supervoxel generation script. 
```
python -m scripts.slic_process_sam infer --image_file xxxx
```
### 3.1.1
问题：polyp标注数大于图片数，但一个图片应该只有一个标注
原因：
```
(ttdg) root@autodl-container-0fc04ca4ba-2313fd84:~/autodl-tmp# 
python /root/autodl-tmp/plpolypdajs.py unique pixel values: 
[ 0 1 2 3 4 5 6 7 8 248 249 250 251 252 253 254 255] connected components (excluding bg): 1 findContours returned: 1
```
掩码不是纯黑白二值图，而是被保存成了灰度渐变或者有轻微压缩误差的图像（比如 JPEG 保存过、或者 PNG 压缩出了一堆伪像）。  

解决方法：强行二值化
```
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
mask = np.where(mask > 128, 255, 0).astype(np.uint8)
```
### 3.1.2

问题:  
fundus的标注数量应该是图片数量两倍，但是多很多

原因：把很小的连通区域也标进去了

解决方法：
忽略小碎片
```
num_labels, labels = cv2.connectedComponents(binm)
area_thresh = binm.size * 0.001  # 小于0.1%像素面积的忽略
counts = 0
for i in range(1, num_labels):
    if np.sum(labels == i) >= area_thresh:
        counts += 1
```

问题：
```
Assertion `t >= 0 && t < n_classes` failed

标签和模型输出类别数不匹配
```
解决方法：
验证集设置错误，改一下验证集

问题：
```
ValueError: Does not validate against any of the Union subtypes
Subtypes: [<class 'NoneType'>, <class 'lightning.pytorch.core.module.LightningModule'>]
Errors:
  - Expected a <class 'NoneType'>
  - 'init_args'
Given value type: <class 'jsonargparse._namespace.Namespace'>
Given value: Namespace(class_path='chest_xray.pl_modules.DomainConditionedCXRLitModule', init_args=Namespace(...))

解决方法：这是用ctrl C中断训练导致权重不完整导致的，完整训练就可以
```
问题：
```
KeyError: np.int64(1)
wandb.log({"conf_mat": confusion_matrix(preds=val_domain_preds, y_true=val_domain_labels, class_names=self.cls_names)})

解决方法：
类别映射那里修改了一下并且如果训练时只有一个域就不画混淆矩阵
```
## 3.2  training stage1和微调
```
export CUDA_VISIBLE_DEVICES=0; python -m scripts.train run --config_file "['configs/train/hyper_parameters_stage1.yaml']"
```
### 报错
### 3.2.1
问题：affine矩阵不匹配
解决方法：因为数据用错了用了4D数据，换数据就好了

## 3.3 training stage4和微调
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nnodes=1 --nproc_per_node=8 -m scripts.train run --config_file "['configs/train/hyper_parameters_stage1.yaml']"
```

```
Evaluation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nnodes=1 --nproc_per_node=8 -m scripts.validation.val_multigpu_point_patch run --config_file "['configs/supported_eval/infer_patch_auto.yaml']" --dataset_name 'xxxx'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nnodes=1 --nproc_per_node=8 -m scripts.validation.val_multigpu_point_patch run --config_file "['configs/supported_eval/infer_patch_point.yaml']" --dataset_name 'xxxx'

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;torchrun --nnodes=1 --nproc_per_node=8 -m scripts.validation.val_multigpu_autopoint_patch run --config_file "['configs/supported_eval/infer_patch_autopoint.yaml']" --dataset_name 'xxxx'
```
## 3.4 Results
指标为Dice score (DSC, %)
| 测试方法 | 主表 | Mine |
| ---- | ---- | ---- |
| auto | 0.952 | 0.9515 |
| auto+point seg | 0.954 | 0.9521
| point only | 0.938 | 0.939 |
