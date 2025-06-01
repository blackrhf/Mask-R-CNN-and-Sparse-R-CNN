在VOC数据集上训练并测试模型 Mask R-CNN 和 Sparse R-CNN 

软件环境:python=3.8/cuda11.8/torch==2.1.0/mmcv2.1.0

配置环境（详细步骤版）

1.首先确保已经安装Anaconda或者Miniconda

2.创建conda环境 conda create --name openmmlab python=3.8 -y

3.启动激活已创建的环境 conda activate openmmlab

4.安装pytorch pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

这里采用 pip 或者 pip3都可以，需要注意的是，这里安装的是cuda11.8
torch==2.1.0 是因为在下一步要与mmcv保持版本对应一致

5.安装MIM，MMEngin和MMCV pip install -U openmim 

mim install mmengine

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

因为在上一步的安装中，我们安装的是 torch==2.1.0版本，这里一步选择 cuda11.8 ，torch 2.1.x 以及后面 mmcv2.1.0，这几个一定保持对应，否则后面再安装完成后测试的时候，会因为版本不对应出错。

6.安装mmdet mim install mmdet

7.启动已经安装好的conda 环境 conda activate openmmlab

--------------------------------------------------------

下载VOC数据集，并把数据集转化为coco格式

python tools/dataset_converters/pascal_voc.py D:\rhf\zhangli\mmdetection-main\data\VOCdevkit -o D:\rhf\zhangli\mmdetection-main\data\output --out-format coco

用的是tools文件夹里面的数据集转化文件，注意输入输出地址根据自己的位置进行相应的更换

训练Mask R-CNN模型：python tools/train.py configs/mask_rcnn/mask-rcnn_r101_fpn_2x_coco.py

选择的是mask-rcnn_r101_fpn_2x_coco模型，注意引用的数据格式源文件，instancecoco.py文件夹里面要做适当的调整，更新类别

训练Sparse R-CNN模型 python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_ms-480-800-3x_coco.py

选择的是sparse-rcnn_r50_fpn_ms-480-800-3x_coco模型，然后同上

把训练log文件转化为tensorboard格式 python showtensorboard.py

注意输入输出文件地址

tensorboard可视化 tensorboard --logdir=work_dirs/mask-rcnn_r101_fpn_2x_coco/20250528_222133/vis_data/log_file

tensorboard --logdir=work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_coco/20250529_145936/vis_data/log_file

可视化Mask R-CNN第一阶段产生的proposal box  python proposalbox.py

代码设置只显示十个box

推理训练集/非训练集内图片 python inference.py

注意输入输出文件地址

test.ipynb完成了在caltech-101数据集上进行微调的分类任务

环境上的配置与sparse-r-cnn一致

jupyter在调试时效果更易观察

每个模块的部分通过markdown进行了功能上的说明

微调性能上都得到了显著的提升，准确率从未进行微调的0.6124都提升至0.9以上

同时用微调的方法保证了算力和运算时间的节省

使用tensorboard进行损失图像的绘制和准确率随epoch增加而变化图像的绘制

结果保存在log日志中





