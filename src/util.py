import datasets
from datasets import Dataset
from trl import ModelConfig
from peft import AutoPeftModelForSeq2SeqLM
from peft.peft_model import PeftModel
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM
from transformers.hf_argparser import DataClassType
from torch.utils.tensorboard.writer import SummaryWriter


import pandas as pd
from typing import Tuple, Optional
import os

import constants
from parser import SFTArguments


def load_radiology_dataset(file_path: str, set_name: str, sanity_check: bool = False, split: bool = False) -> Dataset:
    """Load the custom dataset from a CSV file and convert it to the necessary format.
    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    df = pd.read_csv(file_path)

    if sanity_check:
        df = df.sample(n=min(len(df), 1000), random_state=42)

    dataset = Dataset.from_pandas(df)

    dataset = dataset.rename_columns({"instruction": "prompt",
                                      "chosen_response": "chosen",
                                      "rejected_response": "rejected"})

    if set_name not in ['train', 'test']:
        raise ValueError(
            f"Split must be 'train' or 'test' but received: {set_name}")

    if split:
        dataset: datasets.DatasetDict = Dataset.train_test_split()
        return dataset[set_name]

    return dataset


def load_model_and_tokenizer_dpo(model_config: ModelConfig, args: DataClassType) -> Tuple[PeftModel, Optional[PeftModel], AutoTokenizer]:
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
    return model, model_ref, tokenizer


def load_model_and_tokenizer(args):
    ''' load model and tokenizer '''

    quantization_config = constants.DEFAULT_QUANTIZATION_CONFIG

    model_path = constants.MODELS[args.model] if args.model_path is None else args.model_path
    print(f"[INFO] Loading model at path: {model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                                                  quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_peft_model_and_tokenizer(args):
    ''' load PEFT model and tokenizer '''

    quantization_config = constants.DEFAULT_QUANTIZATION_CONFIG
    model_path = constants.MODELS[args.model] if args.model_path is None else args.model_path
    print(f"[INFO] Loading model at path: {model_path}")
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(
        model_path, quantization_config=quantization_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    return model, tokenizer


def set_preliminaries(train: bool = True):
    ''' parse args, set paths, create dirs, basic checks '''

    # parse arguments, set paths based on expmt params
    parser = HfArgumentParser(SFTArguments)
    args = parser.parse_args_into_dataclasses()[0]
    args.dir_data = os.path.join(constants.DATA_DIR, args.dataset)
    args.dir_models_tuned = os.path.join(
        constants.RESULTS_DIR, "models_tuned", args.dataset, args.model)
    args.dir_out = os.path.join(
        constants.RESULTS_DIR, f"{'dpo_' if args.dpo else ''}output")
    os.makedirs(args.dir_data, exist_ok=True)
    os.makedirs(args.dir_models_tuned, exist_ok=True)
    os.makedirs(args.dir_out, exist_ok=True)

    args.batch_size = constants.DEFAULT_PARAMS['batch_size']
    args.thresh_seq_crop = constants.DEFAULT_PARAMS['thresh_seq_crop']
    assert args.thresh_seq_crop >= 0 and args.thresh_seq_crop < 1
    args.thresh_out_toks = constants.DEFAULT_PARAMS['thresh_out_toks']
    args.n_icl = constants.DEFAULT_PARAMS['n_icl']
    args.use_instruction = constants.DEFAULT_PARAMS['use_instruction']
    args.device = 'cuda:0'

    # SFT args
    if train:
        args.max_trn_epochs = constants.DEFAULT_PARAMS['max_trn_epochs']
        args.n_trn_samples = constants.DEFAULT_PARAMS['n_trn_samples']
        args.n_val_samples = constants.DEFAULT_PARAMS['n_val_samples']
        args.grad_accum_steps = constants.DEFAULT_PARAMS['grad_accum_steps']
        args.lr_num_warmup_steps = constants.DEFAULT_PARAMS['lr_num_warmup_steps']

    # dataset args
    args.summ_task = 'rrs'
    task_dict = constants.PROMPT_COMPONENT[args.summ_task]
    args.prefix = task_dict['prefix']
    args.suffix = task_dict['suffix']
    args.instruction = task_dict['instruction']
    args.n_toks_buffer = 0

    args.case_id = 300

    # create tensorboard dir
    if train:
        dir_tb_log = os.path.join(args.dir_models_tuned, 'logs')
        if not os.path.exists(dir_tb_log):
            os.makedirs(dir_tb_log)

    # init tb writer. via cl: tensorboard --logdir=args.dir_out --port=8888
    writer = SummaryWriter(dir_tb_log) if train else None

    return args, writer
