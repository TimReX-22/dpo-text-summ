from dataclasses import dataclass, field


@dataclass
class DPOArguments:
    beta: float = field(default=0.1, metadata={
                        "help": "the beta parameter for DPO loss"})
    max_length: int = field(default=512, metadata={
                            "help": "max length of each sample"})
    max_prompt_length: int = field(
        default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: int = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    loss_type: str = field(default="sigmoid", metadata={
                           "help": "loss type to use for training"})
    use_ref_model: bool = field(default=False, metadata={
                                "help": "Use a reference model for DPO"})
    sanity_check: bool = field(default=True, metadata={
                               "help": "only train on 1000 samples"})
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "debug argument for distributed training;"
            "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    generate_during_eval: bool = field(
        default=False, metadata={"help": "Generate during evaluation"})
    measure_time: bool = field(default=False, metadata={
                               "help": "measure time of training"})
    model: str = field(default=None, metadata={"help": "model name"})


@dataclass
class SFTArguments:
    model: str = field(default=None, metadata={"help": "model name"})
    dataset: str = field(default=None, metadata={"help": "dataset to use"})
    max_len_abs: int = field(default=512, metadata={
        "help": "max length of sequence"})
    n_samples: int = field(default=None, metadata={
                           "help": "number of samples to evaluate"})
    epoch_eval: int = field(default=None, metadata={
                            "help": "epoch at which to evaluate the model"})
    model_path: str = field(default=None, metadata={
                            "help": "path at which a pretrained model is stored"})
    dpo: bool = field(default=False, metadata={
                      "help": "run inference on a model after DPO"})
