## install 
```
conda create -n nanodet python=3.8 -y
conda activate nanodet
conda install pytorch torchvision torchaudio cudatoolkit=10.2
pip install -r requirements.txt
python train.py config/nanodet-plus-m-1.5x_320.yml > out.file 2>&1 &
```

## 数据集标注格式转换
```
python interhand2coco.py
```
修改需要转换的json文件路径 即可

## 各种运行脚本
1. 测试单张图片demo
```
python demo.py image --config config/nanodet-plus-m-1.5x_320.yml --model ./workspace/nanodet-plus-m-1.5x_320/NanoDet/2022-10-22-13-41-42/checkpoints/epoch=1-step=1626.ckpt --path ./image19336.jpg  --save_result
```
2. 测试一个文件夹中图片
```
python demo.py image --config config/nanodet-plus-m-1.5x_320.yml --model ./workspace/nanodet-plus-m-1.5x_320/NanoDet/2022-10-22-13-41-42/checkpoints/epoch=1-step=1626.ckpt --path ~/dataset/nano_keypoints/images/val/Capture0/ROM04_LT_Occlusion/cam400265  --save_result
```

3. 原始nanodet 手部检测与手部关键点demo
```
python demo_handpose.py image --config config/nanodet-plus-m-1.5x_320.yml --model ./workspace/nanodet-plus-m-1.5x_320/NanoDet/2022-10-22-13-41-42/checkpoints/epoch=1-step=1626.ckpt --path ~/dataset/nano_keypoints/images/val/Capture0/ROM04_LT_Occlusion/cam400265  --save_result
```

4. esnet 为backbone nanodet
```
python demo_handpose.py image --config config/nanodet-esnet_320.yml --model ./workspace/nanodet-esnet_320/model_best/model_best.ckpt --path ~/dataset/nano_keypoints/images/val/Capture0/ROM04_LT_Occlusion/cam400265  --save_result
python demo_handpose.py video --config config/nanodet-esnet_320.yml --model ./workspace/nanodet_model_best.pth --path tests/3.mp4  --save_result

```

## 导出onnx
```
python export_onnx.py --cfg_path config/nanodet-plus-m-1.5x_320.yml --model_path workspace/nanodet-plus-m-1.5x_320/model_last.ckpt --out_path out.onnx --input_shape 320,320
```