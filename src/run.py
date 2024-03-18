# This file is derived from "Clinical Text Summarization by Adapting LLMs | Nature Medicine" at https://github.com/StanfordMIMI/clin-summ.
# Original Authors: Dave Van Veen, Cara Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck, Asad Aali, Christian Bluethgen, Anuj Pareek, Malgorzata Polacin, Eduardo Pontes Reis, Anna Seehofnerova, Nidhi Rohatgi, Poonam Hosamani, William Collins, Neera Ahuja, Curtis P. Langlotz, Jason Hom, Sergios Gatidis, John Pauly, Akshay S. Chaudhari.
# This file has been modified from its original version by Onat Dalmaz, Tim Reinhart and Mike Timmermann.
#
# Citation of the original publication:
# Van Veen, D., Van Uden, C., Blankemeier, L., Delbrouck, J.-B., Aali, A., Bluethgen, C., Pareek, A., Polacin, M., Collins, W., Ahuja, N., Langlotz, C.P., Hom, J., Gatidis, S., Pauly, J., & Chaudhari, A.S. (2024). Adapted Large Language Models Can Outperform Medical Experts in Clinical Text Summarization. Nature Medicine. https://doi.org/10.1038/s41591-024-02855-5. Published 27 February 2024.

import copy
import os
from peft import PeftModel
import time
import torch
from tqdm import tqdm

import util
import process
import summ_dataset


def main():

    # parse arguments. set paths based on expmt params
    args, _ = util.set_preliminaries(train=False)

    # load data
    dataset = summ_dataset.SummDataset(args, task='test')

    # filter out pre-generated samples for this experimental configuration
    for sample in copy.deepcopy(dataset.data):
        if sample['idx'] in dataset.idcs_pregen:
            summ_dataset.remove_sample(dataset.data, sample['idx'])

    # load model, tokenizer
    model, tokenizer = load_model_and_tokenizer_wrapper(args)

    list_out, list_idx = [], []
    t0 = time.time()

    loader = process.get_loader(args, dataset.dataset_obj, tokenizer)

    for step, batch in enumerate(tqdm(loader)):

        # idcs preserve order of input/output
        list_idx.extend(batch['idx'])
        batch = {k: v.to(args.device) for k, v in batch.items()}
        print(f"max_new_tokens: {dataset.max_new_toks}")
        with torch.no_grad():
            output = model.generate(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    do_sample=False,
                                    max_new_tokens=dataset.max_new_toks,
                                    pad_token_id=tokenizer.eos_token_id)

        list_out.extend(tokenizer.batch_decode(output,
                                               skip_special_tokens=True))

    print('generated {} samples for {} expmt in {} sec'.format(
          len(list_idx), args.model, time.time() - t0))

    # add output to dataset, save result.jsonl
    dataset.postprocess_append_output(args, list_idx, list_out)
    dataset.save_data(args, append_pregen=True)


def get_finetuned_model(model, args):
    ''' load model weights which were fine-tuned in-house '''

    if args.epoch_eval == None:  # if not specified, get highest epoch in folder
        subdirs = [ii[0].split('/')[-1]
                   for ii in os.walk(args.dir_models_tuned)]
        epochs_all = [int(ii) for ii in subdirs if ii.isdigit()]
        args.epoch_eval = max(epochs_all)

    dir_model_peft = os.path.join(args.dir_models_tuned, f'{args.epoch_eval}')
    print(f'evaluating model: {dir_model_peft}')
    model = PeftModel.from_pretrained(model, dir_model_peft)
    model.eval()

    return model


def load_model_and_tokenizer_wrapper(args):
    ''' wrapper for loading model and tokenizer '''

    if args.dpo:
        model, tokenizer = util.load_peft_model_and_tokenizer(args)
        tokenizer.eos_token_id = 1
        model.to(args.device)
    else:
        model, tokenizer = util.load_model_and_tokenizer(args)
        tokenizer.eos_token_id = 1
        model = get_finetuned_model(model, args)
        model.to(args.device)

    return model, tokenizer


if __name__ == '__main__':
    main()
