# This file is derived from "Clinical Text Summarization by Adapting LLMs | Nature Medicine" at https://github.com/StanfordMIMI/clin-summ.
# Original Authors: Dave Van Veen, Cara Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck, Asad Aali, Christian Bluethgen, Anuj Pareek, Malgorzata Polacin, Eduardo Pontes Reis, Anna Seehofnerova, Nidhi Rohatgi, Poonam Hosamani, William Collins, Neera Ahuja, Curtis P. Langlotz, Jason Hom, Sergios Gatidis, John Pauly, Akshay S. Chaudhari.
# This file has been modified from its original version by Onat Dalmaz, Tim Reinhart and Mike Timmermann.
#
# Citation of the original publication:
# Van Veen, D., Van Uden, C., Blankemeier, L., Delbrouck, J.-B., Aali, A., Bluethgen, C., Pareek, A., Polacin, M., Collins, W., Ahuja, N., Langlotz, C.P., Hom, J., Gatidis, S., Pauly, J., & Chaudhari, A.S. (2024). Adapted Large Language Models Can Outperform Medical Experts in Clinical Text Summarization. Nature Medicine. https://doi.org/10.1038/s41591-024-02855-5. Published 27 February 2024.


from torch import bfloat16
from peft import LoraConfig
from transformers import BitsAndBytesConfig

DATA_DIR = "./data"
RESULTS_DIR = "./results"

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

##############################################################
### misc filenames ###########################################

FN_INPUTS = 'inputs.csv'
FN_TARGET = 'target.csv'  # target
FN_OUTPUT = 'output.csv'  # generated output
FN_METRICS_JSON = 'metrics.json'
FN_METRICS_TXT = 'metrics.txt'
FN_INP_ICL = 'train.inputs.tok'
FN_TGT_ICL = 'train.target.tok'
FN_IDCS_ICL = 'trn_inputs_index.bin'
FN_RESULT = 'result.jsonl'
FN_TST = 'test'

# keys which should be present when loading data
KEYS_INP = ['idx', 'inputs', 'target']
KEYS_OUT = ['idx', 'inputs', 'target', 'prompt', 'output']
KEYS_TRN = ['input_ids', 'attention_mask', 'labels']

METRICS = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
           'BERT', 'F1-CheXbert', 'F1-Radgraph']  # , 'MEDCON']
MODELS = {"flan-t5-xl": "google/flan-t5-xl"}
DEFAULT_PARAMS = {'n_trn_samples': None,
                  'n_val_samples': None,
                  'batch_size': 6,
                  'grad_accum_steps': 4,
                  'lr0': 1e-3,
                  'lr_schedule': 'linear_decay',
                  'lr_num_warmup_steps': 100,
                  'max_trn_epochs': 3,
                  'n_icl': 0,  # number of in-context examples
                  'use_instruction': True,  # include prefix instruction
                  # only relevant for datasets w modalities (iii)
                  'modalities': 'all',
                  'thresh_seq_crop': 0,  # threshold to remove longest X% of sequences
                  'thresh_out_toks': 0, }

# instructions
INSTRUCTION_RRS = (
    'summarize the radiology report findings into an impression'
    ' with minimal text'
)
INSTRUCTION_CHQ = (
    'summarize the patient health query into one question'
    ' of 15 words or less'
)
INSTRUCTION_PLS = (
    'based on the progress note, generate a list of 3-7 problems'
    ' (a few words each) ranked in order of importance'
)
INSTRUCTION_D2N = (
    'summarize the patient/doctor dialogue into an assessment and plan'
)

# define task-dependent prompt components
PROMPT_COMPONENT = {
    'rrs': {
        'prefix': 'finding',
        'suffix': 'impression',
        'instruction': INSTRUCTION_RRS,
    },
    'chq': {
        'prefix': 'query',
        'suffix': 'summarized question',
        'instruction': INSTRUCTION_CHQ,
    },
    'pls': {
        'prefix': 'progress note',
        'suffix': 'problem list',
        'instruction': INSTRUCTION_PLS,
    },
    'd2n': {
        # 'prefix': 'patient/provider dialogue',
        'prefix': 'patient/doctor dialogue',
        'suffix': 'assessment and plan',
        'instruction': INSTRUCTION_D2N,
    },
}

# datasets which have been implemented to date
DATASETS_IMPLEMENTED = ['iii', 'chq', 'pls', 'opi', 'cxr', 'd2n']

# datasets of radiology reports w modality subdirs
DATASETS_W_MODALITIES = ['iii']

PATIENCE = 5
