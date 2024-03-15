# This file is derived from "Clinical Text Summarization by Adapting LLMs | Nature Medicine" at https://github.com/StanfordMIMI/clin-summ.
# Original Authors: Dave Van Veen, Cara Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck, Asad Aali, Christian Bluethgen, Anuj Pareek, Malgorzata Polacin, Eduardo Pontes Reis, Anna Seehofnerova, Nidhi Rohatgi, Poonam Hosamani, William Collins, Neera Ahuja, Curtis P. Langlotz, Jason Hom, Sergios Gatidis, John Pauly, Akshay S. Chaudhari.
#
# This file has been modified from its original version by Onat Dalmaz, Tim Reinhart and Mike Timmermann.
#
# Citation of the original publication:
# Van Veen, D., Van Uden, C., Blankemeier, L., Delbrouck, J.-B., Aali, A., Bluethgen, C., Pareek, A., Polacin, M., Collins, W., Ahuja, N., Langlotz, C.P., Hom, J., Gatidis, S., Pauly, J., & Chaudhari, A.S. (2024). Adapted Large Language Models Can Outperform Medical Experts in Clinical Text Summarization. Nature Medicine. https://doi.org/10.1038/s41591-024-02855-5. Published 27 February 2024.

import math
import os
import peft
import time
import torch
from tqdm.autonotebook import tqdm
import transformers

import constants
import util
import process
from summ_dataset import SummDataset


def main():

    # set preliminaries, model, tokenizer

    args, writer = util.set_preliminaries()
    model, tokenizer = util.load_model_and_tokenizer(args)
    model = get_tunable_model(model, args)

    # load data
    trn_dataset = SummDataset(args, task='trn').dataset_obj
    trn_loader = process.get_loader(args, trn_dataset, tokenizer)
    val_dataset = SummDataset(args, task='trn').dataset_obj
    val_loader = process.get_loader(args, val_dataset, tokenizer)
    args.steps_per_epoch = len(trn_loader)
    print(f'{len(trn_dataset)} samples w batch size {args.batch_size}, '
          f'hence {args.steps_per_epoch} gradient steps per epoch')

    # define optimizer, lr scheduler
    num_trn_steps = len(trn_loader) * args.max_trn_epochs
    optimizer, lr_scheduler = define_optimizer(args, model, num_trn_steps)

    model.train()
    best_val_loss = math.inf
    # early stop if loss doesn't reach new min in consec epochs
    patience = constants.PATIENCE
    n_steps = 0  # track number of steps taken
    trn_losses = []
    print('begin training!')

    start_time = time.time()

    for epoch in range(args.max_trn_epochs):
        with tqdm(total=len(trn_loader)) as pbar:  # progress bar
            for idx_b, batch in enumerate(trn_loader):
                n_steps += 1

                # forward pass
                batch = prep_batch(args, batch)
                outputs = model(**batch)

                # compute loss, gradient step
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()

                # optimizer step/zero after grad_accum_steps steps
                if (n_steps % args.grad_accum_steps == 0) or (n_steps == len(trn_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

                detached_loss = loss.detach().float()
                trn_losses.append(detached_loss)
                writer.add_scalar('trn_loss', detached_loss, n_steps)
                writer.add_scalar('trn_perplexity',
                                  torch.exp(detached_loss), n_steps)
                pbar.update(1)

        # calculate validation loss
        with tqdm(total=len(val_loader)) as pbar:  # progress bar
            val_losses = []
            for batch in val_loader:
                batch = prep_batch(args, batch)
                with torch.no_grad():
                    outputs_val = model(**batch)
                    val_losses.append(outputs_val.loss.detach().float())
                pbar.update(1)

        trn_loss_epoch = sum(trn_losses) / len(trn_losses)
        val_loss_epoch = sum(val_losses) / len(val_losses)
        trn_perplexity_epoch = torch.exp(trn_loss_epoch)
        val_perplexity_epoch = torch.exp(val_loss_epoch)

        writer.add_scalar('lr', lr_scheduler.get_lr()[0], epoch)
        writer.add_scalar('trn_loss_epoch', trn_loss_epoch, epoch)
        writer.add_scalar('val_loss_epoch', val_loss_epoch, epoch)
        writer.add_scalar('trn_perplexity_epoch', trn_perplexity_epoch, epoch)
        writer.add_scalar('val_perplexity_epoch', val_perplexity_epoch, epoch)

        print(f"epoch: {epoch}/{args.max_trn_epochs}, "
              f"trn_loss_epoch: {trn_loss_epoch}, "
              f"trn_perplexity_epoch: {trn_perplexity_epoch}, "
              f"val_loss_epoch: {val_loss_epoch}, "
              f"val_perplexity_epoch: {val_perplexity_epoch}, "
              f"lr: {lr_scheduler.get_lr()[0]}")

        # save model at each epoch
        model_save_dir = os.path.join(args.dir_models_tuned, f'{epoch}')
        model.save_pretrained(model_save_dir)
        # saving the tokenizer is required for DPO
        tokenizer.save_pretrained(model_save_dir)

        # early stopping
        if val_loss_epoch > best_val_loss:
            if patience == 0:
                print(f'stopping early at epoch {epoch}!')
                break
            else:
                patience -= 1
        else:
            patience = constants.PATIENCE
            best_val_loss = val_loss_epoch

    end_time = time.time()
    execution_time = round(end_time - start_time, 4)
    print(f"Ran SFT in {execution_time:.4f} seconds")


def define_optimizer(args, model, num_trn_steps):
    ''' given parameters
        define optimizer '''

    # extract learning rate params
    case = constants.DEFAULT_PARAMS
    lr0 = case['lr0']  # initial learning rate

    # define optimizer, lr_scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=lr0,
                                   no_deprecation_warning=True)

    if case['lr_schedule'] == 'polynomial_decay':

        lrn = case['lrn']  # final learning rate
        lr_decay_power = case['lr_decay_power']  # rate of polynomial decay
        str_ = f'using polynomial decay scheduler with lr0 {lr0}, '
        str_ += f'lrn {lrn}, power {lr_decay_power},'

        lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            lr_end=lrn,
            power=lr_decay_power,
            num_warmup_steps=args.lr_num_warmup_steps,
            num_training_steps=num_trn_steps,
        )

    elif case['lr_schedule'] == 'linear_decay':

        str_ = f'using linear scheduler with lr0 {lr0},'
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_num_warmup_steps,
            num_training_steps=num_trn_steps,
        )

    elif case['lr_schedule'] == 'constant':

        str_ = f'using constant learning rate {lr0},'
        lr_scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_num_warmup_steps,
        )

    else:
        raise NotImplementedError('learning rate method not implemented')

    str_ += f' and {args.lr_num_warmup_steps} warm-up steps!'
    print(str_)

    return optimizer, lr_scheduler


def get_tunable_model(model, args):
    ''' prep model for param-efficient fine-tuning '''

    # prepare for k-bit training
    model = peft.prepare_model_for_kbit_training(model)

    # get peft configs based on architecture (task_type) and fine-tuning method
    config = constants.DEFAULT_LORA_CONFIG
    print(f"LoRA config: {config}")

    # wrap model w peft configs
    model = peft.get_peft_model(model, config).to(args.device)
    model.print_trainable_parameters()

    return model


def prep_batch(args, batch):
    ''' remove irrelevant dict keys needed for training
        move to device '''

    for key in list(batch.keys()):
        if key not in constants.KEYS_TRN:
            batch.pop(key)
    batch = {k: v.to(args.device) for k, v in batch.items()}

    return batch


if __name__ == '__main__':
    main()
