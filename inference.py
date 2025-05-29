import mmdet
from mmdet.apis import DetInferencer

# 显式指定 config（建议这样做）
config_path = 'work_dirs/sparse_rcnn/sparse-rcnn_r50_fpn_ms-480-800-3x_coco.py'

inferencer = DetInferencer(
    weights='work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_coco/epoch_14.pth',
    # config=config_path,  # 明确指定 config，避免 registry 警告
    device='cuda:0',
    # save_dir='outputs/logs'  # 让可视化后端正常运行，避免日志警告
)

# 推理输出路径，可视化图会保存在 out_dir 里
inferencer('demo/out', out_dir='demo/outputs/outmask', no_save_pred=False)
inferencer('demo/in', out_dir='demo/outputs/inmask', no_save_pred=False)
