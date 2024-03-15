import constants
import util
from parser import DPOArguments
from peft import AutoPeftModelForSeq2SeqLM
from transformers import HfArgumentParser, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, ModelConfig
from tqdm import tqdm
import time
import os
import pandas as pd
tqdm.pandas()


def run_dpo():
    parser = HfArgumentParser((DPOArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()

    model = AutoPeftModelForSeq2SeqLM.from_pretrained(
        model_config.model_name_or_path,
        quantization_config=constants.DEFAULT_QUANTIZATION_CONFIG,
        is_trainable=True)

    model_ref = AutoPeftModelForSeq2SeqLM.from_pretrained(
        model_config.model_name_or_path,
        quantization_config=constants.DEFAULT_QUANTIZATION_CONFIG) if args.use_ref_model else None
    print(
        f"[INFO] Using {'NO' if not args.use_ref_model else 'a'} reference model")

    print(f"[INFO] Model loaded: {model_config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_path = os.path.join(
        constants.DATA_DIR, "radiology_report_comparison_dataset.csv")
    train_dataset = util.load_radiology_dataset(
        dataset_path, "train", sanity_check=args.sanity_check)
    eval_dataset = util.load_radiology_dataset(
        dataset_path, "test", sanity_check=args.sanity_check)

    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        max_prompt_length=args.max_prompt_length,
        generate_during_eval=args.generate_during_eval,
        loss_type=args.loss_type,
    )

    if args.measure_time:
        start_time = time.time()

    trainer.train()

    if args.measure_time:
        end_time = time.time()
        execution_time = start_time - end_time
        print(f"[INFO] Execution time: {execution_time}")

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    run_dpo()
