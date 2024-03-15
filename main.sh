RUN_DPO=true
RUN_SFT=false

if $RUN_DPO; then
    python src/dpo.py \
        --model flan-t5-xl \
        --model_name_or_path='./results/models_tuned/opi/flan-t5-xl/0' \
        --output_dir="" \
        --per_device_train_batch_size 4 \
        --max_steps 100 \
        --learning_rate 0.00001 \
        --gradient_accumulation_steps 2 \
        --logging_steps 10 \
        --eval_steps 1000 \
        --use_ref_model false \
        --optim rmsprop \
        --loss_type=sigmoid \
        --warmup_steps 100 \
        --max_length 512 \
        --max_prompt_length 128 \
        --max_target_length 128 \
        --load_in_4bit \
        --bf16 \
        --logging_first_step \
        --no_remove_unused_columns \
        --use_peft \
        --lora_r 16 \
        --lora_alpha 16 \
        --measure_time

    python src/run.py --model flan-t5-xl \
        --model_path="./results/dpo_output/flan-t5-xl" \
        --dataset opi \
        --n_samples 250 \
        --dpo

    python src/calc_metrics.py --model flan-t5-xl \
        --dataset opi \
        --n_samples 250 \
        --dpo
fi

if $RUN_SFT; then
    python src/sft.py --model flan-t5-xl --dataset opi

    python src/run.py --model flan-t5-xl \
        --dataset opi \
        --n_samples 250

    python src/calc_metrics.py --model flan-t5-xl \
        --dataset opi \
        --n_samples 250
fi
