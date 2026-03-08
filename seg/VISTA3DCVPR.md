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
