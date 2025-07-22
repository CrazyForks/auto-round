import os
import argparse
import tqdm

# ensure that unnecessary memory is released during quantization.
os.environ.setdefault("PT_HPU_WEIGHT_SHARING", "0")
if int(os.getenv("WORLD_SIZE", "0")) > 0:
    os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
    os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")


import torch
import habana_frameworks.torch.core as htcore

from neural_compressor.torch.quantization import (
    FP8Config,
    prepare,
    convert,
    finalize_calibration,
    save,
    load,
)
from neural_compressor.torch.utils import get_used_hpu_mem_MB, get_used_cpu_mem_MB, logger, forward_wrapper
from neural_compressor.torch.utils.block_wise import block_wise_calibration
from neural_compressor.torch.utils.llm_utility import (
    initialize_model_and_tokenizer,
    get_default_llm_dataloader,
    llm_benchmark,
)

# use no_grad mode for quantization
htcore.hpu_set_env()
hpu_mem_0 = get_used_hpu_mem_MB()
cpu_mem_0 = get_used_cpu_mem_MB()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana FP8 quantization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name or path")
    parser.add_argument("--quantize", action="store_true", help="whether to quantize model")
    parser.add_argument("--autoround", action="store_true", help="whether to autoround model")
    parser.add_argument("--iters", default=None, type=int, help="iters for autoround.")
    parser.add_argument("--seqlen", default=None, type=int, help="sequence length for autoround.")
    parser.add_argument("--nsamples", default=None, type=int, help="number of samples for autoround.")
    parser.add_argument("--scale_method", type=str, default="maxabs_hw", help="Choose scale method", choices=[
        # per-tensor
        "unit_scale", "hw_aligned_single_scale", "maxabs_hw", "maxabs_pow2", 
        "maxabs_arbitrary", "maxabs_hw_opt_weight", "maxabs_pow2_opt_weight", 
        # per-channel
        "act_maxabs_hw_weights_pcs_maxabs_pow2", "act_maxabs_hw_weights_pcs_opt_pow2", 
        "act_maxabs_pow2_weights_pcs_maxabs_pow2", "act_maxabs_pow2_weights_pcs_opt_pow2",
    ])
    parser.add_argument("--use_hpu_graph", action="store_true", help="whether to use hpu graph mode to accelerate performance")
    parser.add_argument("--enable_block_wise_calibration", action="store_true", help="whether to use block-wise calibration")
    parser.add_argument("--disable_optimum_habana", action="store_true", help="whether to use adapt_transformers_to_gaudi")
    parser.add_argument("--save", action="store_true", help="whether to save the quantized model")
    parser.add_argument("--load", action="store_true", help="whether to load the quantized model")
    parser.add_argument("--save_path", type=str, default="saved_results", help="path to save the quantized model")
    parser.add_argument("--accuracy", action="store_true", help="accuracy measurement")
    parser.add_argument("--performance", action="store_true", help="performance measurement")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size for accuracy measurement.")
    parser.add_argument("--num_fewshot", default=0, type=int, help="num_fewshot of lm_eval.")
    parser.add_argument(
        "--mxfp8_mod_list", 
        type=str, 
        nargs="+",  # 接受一个或多个字符串作为列表
        default=[],  # 默认值
        help="List of module names or patterns for MXFP8 quantization."
    )
    parser.add_argument(
        "--fp8_mod_list", 
        type=str, 
        nargs="+",  # 接受一个或多个字符串作为列表
        default=[],  # 默认值
        help="List of module names or patterns for MXFP8 quantization."
    )
    parser.add_argument("--dump_stats_path", type=str, default="./hqt_output/measure", help="path and prefix to calibration info file.")
    parser.add_argument("--tasks", default="piqa,winogrande,hellaswag,lambada_openai,mmlu",
                        type=str, help="tasks for accuracy validation, text-generation and code-generation tasks are different.")
    parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-10k", help="dataset name for calibration dataloader")
    parser.add_argument("--limit", type=int, default=None, help="number of samples for accuracy evaluation")
    args = parser.parse_args()

    model, tokenizer = initialize_model_and_tokenizer(args.model_name_or_path, use_load=args.load, device="hpu")
    # show used memory
    logger.info(f"After loading model, used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
    logger.info(f"After loading model, used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")

    if args.quantize:
        print("mxfp8:", args.mxfp8_mod_list)
        print("fp8:", args.fp8_mod_list)
        from neural_compressor.torch.experimental.fp4.quantize import qdq_model
        model = qdq_model(
            model, 
            dtype="mxfp4", 
            mxfp8_mod_list=args.mxfp8_mod_list,
            fp8_mod_list=args.fp8_mod_list,
        )
        print(model)
        # show used memory
        logger.info(f"Used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")

    if args.autoround:
        from auto_round import AutoRound
        from neural_compressor.torch.experimental.fp4.quantize import MXFP4_MODULE_MAPPING, match_pattern

        layer_config = {}
        fp8_config = {
            "data_type": "fp8",
            "act_data_type": "fp8",
        }
        mxfp4_config = {
            "group_size": 32,
            "data_type": "mx_fp8",
            "act_data_type": "mx_fp8",
        }
        mxfp8_config = {
            "group_size": 32,
            "data_type": "mx_fp8",
            "act_data_type": "mx_fp8",
        }
        module_name_to_quantize: list[str] = [
            n for n, m in model.named_modules() if \
                isinstance(m, tuple(MXFP4_MODULE_MAPPING.keys()))
        ]
        for name in module_name_to_quantize:
            if match_pattern(name, args.mxfp8_mod_list):
                layer_config.update({name: mxfp8_config})
            if match_pattern(name, args.fp8_mod_list):
                layer_config.update({name: fp8_config})
        print(layer_config.keys())
        if not "lm_head" in layer_config:
            # lm_head is not supported by MXFP8, so we use fp8 for it.
            layer_config["lm_head"] = mxfp4_config

        from auto_round import AutoRound
        autoround = AutoRound(
            model, 
            tokenizer,
            low_gpu_mem_usage=True,
            device="hpu",
            group_size=32,
            data_type="mx_fp4",
            act_data_type="mx_fp4",
            layer_config=layer_config,
        )
        autoround.quantize()
        model = autoround.model
        print(model)
        
        # show used memory
        logger.info(f"Used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")


    if args.model_name_or_path == "meta-llama/Meta-Llama-3.1-70B-Instruct":
        ds_inference_kwargs = {
            "dtype": torch.bfloat16,
            "tensor_parallel": {"tp_size": 4},
        }
        import deepspeed

        ds_model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = ds_model.module

    # preprocess model for accuracy and performance measurement
    if not args.load and not args.autoround and not args.quantize:
        # compare fp8 with bf16, not fp32.
        model = model.to(torch.bfloat16)
    model = model.eval().to("hpu")
    # show used memory
    logger.info(f"Totally used HPU memory: {round(get_used_hpu_mem_MB()/1024, 3)} GiB")
    logger.info(f"Totally used CPU memory: {round(get_used_cpu_mem_MB()/1024, 3)} GiB")
    if args.use_hpu_graph:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        model = wrap_in_hpu_graph(model)
    htcore.hpu_inference_initialize(model, mark_only_scales_as_const=True)
    # show used memory
    logger.info(f"Totally used HPU memory: {round(get_used_hpu_mem_MB()/1024, 3)} GiB")
    logger.info(f"Totally used CPU memory: {round(get_used_cpu_mem_MB()/1024, 3)} GiB")

    if args.accuracy:
        from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
        print(model)
        eval_args = LMEvalParser(
            model="hf", 
            user_model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            tasks=args.tasks,
            device="hpu",
            pad_to_buckets=True,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            add_bos_token=True,
        )
        results = evaluate(eval_args)
        torch.hpu.synchronize()
        all_accuracy = {}
        for task_name, task_results in results["results"].items():
            if task_name in ["hellaswag", "lambada_openai", "piqa", "winogrande", "mmlu"]:
                accu = task_results['acc,none']
                all_accuracy[task_name] = accu
                print(f"Accuracy for {task_name}: {accu:.4f}")
        print(f"Overall accuracy: {sum(all_accuracy.values())/len(all_accuracy):.4f}")

        # show used memory
        logger.info(f"Used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")


    if args.performance:
        llm_benchmark(model, args.batch_size, args.seq_len)
        # show used memory
        logger.info(f"Used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")

