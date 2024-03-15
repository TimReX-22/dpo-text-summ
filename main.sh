RUN_DPO=true

if $RUN_DPO; then
    python src/dpo.py \
        --model_name_or_path='/dataNAS/people/onat/reward_model/Project/rlhf_checkpoints/flan-t5-xl_no_dpo' \
        --per_device_train_batch_size 4 \
        --max_steps 500 \
        --learning_rate 0.00001 \
        --gradient_accumulation_steps 2 \
        --logging_steps 10 \
        --eval_steps 1000 \
        --use_ref_model false \
        --output_dir='/dataNAS/people/onat/reward_model/Project/output' \
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
fi
