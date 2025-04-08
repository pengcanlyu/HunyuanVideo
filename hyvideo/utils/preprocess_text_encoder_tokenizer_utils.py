import argparse
import torch
import gc
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)


def preprocess_text_encoder_tokenizer(args):
    # 清理CUDA缓存以释放内存
    torch.cuda.empty_cache()
    gc.collect()

    processor = AutoProcessor.from_pretrained(args.input_dir)
    # 使用 device_map="auto" 自动分配模型到可用的设备上
    model = LlavaForConditionalGeneration.from_pretrained(
        args.input_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"  # 自动将模型分配到可用的设备上
    )

    model.language_model.save_pretrained(
        f"{args.output_dir}"
    )
    processor.tokenizer.save_pretrained(
        f"{args.output_dir}"
    )
    
    # 清理内存
    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The path to the llava-llama-3-8b-v1_1-transformers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output path of the llava-llama-3-8b-text-encoder-tokenizer."
        "if '', the parent dir of output will be the same as input dir.",
    )
    args = parser.parse_args()

    if len(args.output_dir) == 0:
        args.output_dir = "/".join(args.input_dir.split("/")[:-1])

    preprocess_text_encoder_tokenizer(args)
