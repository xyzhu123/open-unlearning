#!/bin/bash

wandb_project=""
model_name="hubble-8b-500b_toks-perturbed-hf"
model_name_or_path="/path/to/model"

# Fixed parameters
lorra_alpha=10
layers="10,20"
transform_layers="-1"

# hubble-500b
# bash_key="hubble-500b"
# forget_topics=(hubble_mmlu-train-unlearn-256)
# retain_topic="wikitext-train"
# lrs=(5e-4 1e-4 5e-5)
# random_seeds=(42)
# max_steps_list=(142 284) # # 2/4 times: mmlu/gtbg-64: (286 572), mmlu/gtbg-256: (142 284)

# hubble-500b-wikitext
# bash_key="hubble-500b-wikitext"
# forget_topics=(hubble_gutenberg-train-unlearn-256)
# retain_topic="wikitext-train"
# lrs=(1e-3 5e-4 5e-5)
# random_seeds=(42)
# max_steps_list=(568) # # 8 times: mmlu/gtbg-64: (1144), mmlu/gtbg-256: (568)

# hubble-500b-keep
# bash_key="hubble-500b-keep"
# forget_topics=(hubble_mmlu-train-unlearn-256)
# retain_topic="hubble_mmlu-train-keep-256"
# lrs=(1e-3 5e-4 5e-5)
# random_seeds=(42)
# max_steps_list=(568) # # 8 times: mmlu/gtbg-64: (1144), mmlu/gtbg-256: (568)

# hubble-500b-keep
# bash_key="hubble-500b-keep"
# forget_topics=(hubble_gutenberg-train-unlearn-256)
# retain_topic="hubble_gutenberg-train-keep-256"
# lrs=(1e-3 5e-4 5e-5)
# random_seeds=(42)
# max_steps_list=(572) # # 4 times: mmlu/gtbg-64: (572), mmlu/gtbg-256: (284)

# hubble-500b-wikitext
# bash_key="hubble-500b-wikitext"
# forget_topics=(hubble_mmlu-train-unlearn-64)
# retain_topic="wikitext-train"
# lrs=(1e-3 5e-4 5e-5)
# random_seeds=(42)
# max_steps_list=(572) # # 4 times: mmlu/gtbg-64: (572), mmlu/gtbg-256: (284)

eval "$(conda shell.bash hook)"
for forget_topic in "${forget_topics[@]}"; do
    for max_steps in "${max_steps_list[@]}"; do
        for lr in "${lrs[@]}"; do
            for random_seed in "${random_seeds[@]}"; do
                run_name="CB_${forget_topic}_${model_name}_${bash_key}_${random_seed}_${lr}"
                output_dir="/path/to/output/${run_name}"
                
                echo "Running with lr=$lr, max_steps=$max_steps, random_seed=$random_seed"
                echo "model_name_or_path=$model_name_or_path"
                echo "output_dir=$output_dir"

                conda activate cb

                output=$(accelerate launch --config_file configs/accelerate_zero1.yaml \
                    --num_processes 1 --main_process_port $MASTER_PORT --deepspeed_hostfile ds_hostfile \
                        src/lorra_circuit_breaker.py \
                            --model_name_or_path $model_name_or_path \
                            --target_layers $layers \
                            --transform_layers $transform_layers \
                            --lorra_alpha $lorra_alpha \
                            --lora_r 16 \
                            --lora_alpha 16 \
                            --lora_dropout 0.05 \
                            --output_dir $output_dir \
                            --overwrite_output_dir \
                            --max_steps $max_steps \
                            --bf16 True \
                            --per_device_train_batch_size 8 \
                            --per_device_eval_batch_size 32 \
                            --gradient_accumulation_steps 1 \
                            --do_eval \
                            --evaluation_strategy "steps" \
                            --eval_steps 1000 \
                            --save_total_limit 0 \
                            --learning_rate $lr \
                            --weight_decay 0. \
                            --lr_scheduler_type "constant" \
                            --logging_strategy "steps" \
                            --logging_steps 1 \
                            --tf32 True \
                            --model_max_length 8192 \
                            --q_lora False \
                            --gradient_checkpointing True \
                            --report_to wandb \
                            --log_every 1 \
                            --wandb_project $wandb_project \
                            --bash_key $bash_key \
                            --forget_topic $forget_topic \
                            --retain_topic $retain_topic \
                            --random_seed $random_seed 2>&1)

                last_line=$(echo "$output" | tail -n 1)
                echo "Finished running with lr=$lr, random_seed=$random_seed"
                echo "$output"

                # Check if the last line contains a wandb ID (updated pattern)
                if [[ $last_line == *"wandb:"* ]]; then
                    # Extract the run_id (now matching the actual format)
                    run_id=$(echo $last_line | awk '{print $NF}')
                        
                    echo "Starting lm-eval for wandb run_id: $run_id"
                    conda activate hubble_eval
                    
                    lm-eval --model vllm \
                            --model_args pretrained=${output_dir},trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.8,enforce_eager=True \
                            --tasks hubble_mmlu,tinyMMLU,tinyWinogrande,tinyHellaswag,wikitext \
                            --batch_size auto \
                            --wandb_args project=$wandb_project,id=$run_id,resume=allow

                    # Remove the model directory after successful evaluation
                    echo "Removing model directory: ${output_dir}"
                    rm -r "${output_dir}"
                else
                    echo "ERROR: Training failed, skipping lm-eval."
                fi
            done
        done
    done
done