import os
import time
import tempfile
from pathlib import Path
from loguru import logger
from datetime import datetime
import pyzipper  # 加密压缩包的库

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # 创建保存文件夹
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # 加载模型
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # 获取更新后的参数
    args = hunyuan_video_sampler.args

    # 开始采样
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # 保存样本到加密压缩包
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        zip_path = f"{save_path}/{time_flag}_hunyuan_videos.zip"
        
        # 尝试使用RAM磁盘（如果可用），否则回退到临时目录
        temp_base_dir = "/dev/shm" if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
        
        # 创建临时目录，优先使用RAM内存
        with tempfile.TemporaryDirectory(dir=temp_base_dir) as temp_dir:
            try:
                # 将视频保存到临时目录
                video_paths = []
                for i, sample in enumerate(samples):
                    sample = samples[i].unsqueeze(0)
                    video_filename = f"seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
                    temp_video_path = os.path.join(temp_dir, video_filename)
                    
                    # 保存视频到临时目录
                    save_videos_grid(sample, temp_video_path, fps=24)
                    video_paths.append((temp_video_path, video_filename))
                    logger.info(f'临时视频已创建（将被添加到加密压缩包中）')
                
                # 创建加密压缩包
                with pyzipper.AESZipFile(zip_path, 'w', compression=pyzipper.ZIP_LZMA, encryption=pyzipper.WZ_AES) as zipf:
                    # 设置密码
                    zipf.setpassword(b'1234')  # 密码设为1234
                    
                    # 将每个视频添加到压缩包（使用writestr避免存储路径元数据）
                    for video_path, video_filename in video_paths:
                        with open(video_path, 'rb') as f:
                            zipf.writestr(video_filename, f.read())
                        logger.info(f'已将视频"{video_filename}"添加到加密压缩包')
                
                logger.info(f'所有视频已保存到加密压缩包: {zip_path} (密码: 1234)')
            
            except Exception as e:
                logger.error(f"创建加密压缩包时出错: {e}")
                raise
            
            # 临时目录将在此块结束后自动清理

if __name__ == "__main__":
    main()
