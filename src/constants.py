from torch import bfloat16
from peft import LoraConfig
from transformers import BitsAndBytesConfig

DATA_DIR = "./data"

DEFAULT_LORA_CONFIG = LoraConfig(task_type="SEQ_2_SEQ_LM",
                                 inference_mode=False,
                                 r=8,
                                 lora_alpha=32,
                                 lora_dropout=0.1)

DEFAULT_QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bfloat16
)
