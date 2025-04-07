import os
import time
import uuid
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
    
    # 创建保存文件夹（只用于存放最终的加密压缩包）
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
        
        # 使用/dev/shm作为安全临时目录（已通过测试确认可用）
        # 使用唯一ID避免并发冲突
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = f"/dev/shm/hunyuan_temp_{unique_id}"
        
        try:
            # 创建临时目录
            os.makedirs(temp_dir, exist_ok=True)
            logger.info(f"已创建安全临时目录: {temp_dir}")
            
            # 将视频保存到临时目录
            video_paths = []
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                video_filename = f"seed{outputs['seeds'][i]}_{outputs['prompts'][i][:50].replace('/','_').replace(' ','_')}.mp4"
                temp_video_path = os.path.join(temp_dir, video_filename)
                
                # 保存视频到临时目录
                save_videos_grid(sample, temp_video_path, fps=24)
                video_paths.append((temp_video_path, video_filename))
                logger.info(f'临时视频已创建: {video_filename}')
            
            # 创建加密压缩包
            with pyzipper.AESZipFile(zip_path, 'w', compression=pyzipper.ZIP_LZMA, encryption=pyzipper.WZ_AES) as zipf:
                # 设置密码
                zipf.setpassword(b'1234')  # 密码设为1234
                
                # 将每个视频添加到压缩包
                for video_path, video_filename in video_paths:
                    with open(video_path, 'rb') as f:
                        zipf.writestr(video_filename, f.read())
                    # 立即删除临时视频文件
                    os.remove(video_path)
                    logger.info(f'已将视频添加到加密压缩包并删除临时文件: {video_filename}')
            
            logger.info(f'所有视频已保存到加密压缩包: {zip_path} (密码: 1234)')
            
        except Exception as e:
            logger.error(f"创建加密压缩包时出错: {e}")
            raise
            
        finally:
            # 无论成功与否，都尝试清理临时目录
            try:
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(temp_dir)
                logger.info(f"已清理临时目录: {temp_dir}")
            except Exception as e:
                logger.error(f"清理临时目录时出错: {e}")

if __name__ == "__main__":
    main()
