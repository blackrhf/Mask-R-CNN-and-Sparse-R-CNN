from torch.utils.tensorboard import SummaryWriter
import json
import glob
import os

def main(base_dir):
    # 创建对应的 TensorBoard writer
    train_writer = SummaryWriter(os.path.join(base_dir, 'log_file/train'))
    test_writer = SummaryWriter(os.path.join(base_dir, 'log_file/test'))
    eval_writer = SummaryWriter(os.path.join(base_dir, 'log_file/eval'))

    # 查找所有的 .json 文件
    json_files = glob.glob(os.path.join(base_dir, '*.json'))
    if not json_files:
        print("No JSON files found.")
        return

    json_file = json_files[0]
    print(f"Processing file: {json_file}")

    with open(json_file, 'r') as f:
        for line in f:
            try:
                j = json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # 跳过无效行

            # 判断是训练还是评估数据
            if 'mode' in j and j['mode'] == 'test':
                writer = test_writer
                global_step = j.get('step', j.get('iter'))  # 使用 step 或 fallback 到 iter
                prefix = ''
            elif 'pascal_voc/mAP' in j:
                writer = eval_writer
                global_step = j.get('step')
                prefix = 'val/'
            else:
                # 假设是验证数据
                writer = train_writer
                global_step = j.get('step', j.get('iter'))
                prefix = 'val/'

            # 写入所有数值型字段
            for key, value in j.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(prefix + key, value, global_step=global_step)

    train_writer.close()
    test_writer.close()
    eval_writer.close()
    print("TensorBoard logs written successfully.")

if __name__ == '__main__':
    base_dir = 'work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_coco/20250529_145936/vis_data'
    # base_dir = 'work_dirs/mask-rcnn_r101_fpn_2x_coco/20250528_222133/vis_data'
    main(base_dir)