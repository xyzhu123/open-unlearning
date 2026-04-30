#!/bin/bash

unlearn_method="rmu"
model_name="hubble-8b-500b_toks-perturbed-hf"
model_name_or_path="path/to/model"

# # hubble-separated
# bash_key="hubble-separated"
# wandb_project=""
# alphas=(100 1000 10000)
# steering_coeffs_list=(5 50 500)
# lrs=(5e-5 3e-5 1e-5)
# batch_size=1
# gradient_accumulation_steps=4
# max_unlearn_steps_list=(1000) # make sure to enumerate the whole dataset
# random_seeds=(42) 
# max_len=512


# # hubble-500b
# forget_topics=(hubble_mmlu-train-unlearn-64)
# retain_topic="wikitext-train"
# bash_key="hubble-500b"
# wandb_project=""
# alphas=(100 1000 10000)
# steering_coeffs_list=(5 50 500)
# lrs=(5e-5 1e-5)
# batch_size=1
# gradient_accumulation_steps=4
# max_unlearn_steps_list=(572) # 4 times: mmlu/gtbg-64: 572, mmlu/gtbg-256: 284
# random_seeds=(42) 
# max_len=512

# hubble-500b-wikitext
# forget_topics=(hubble_gutenberg-train-unlearn-256)
# retain_topic="wikitext-train"
# bash_key="hubble-500b"
# wandb_project=""
# alphas=(100 1000 10000)
# steering_coeffs_list=(5 50 500)
# lrs=(5e-5 1e-5)
# batch_size=1
# gradient_accumulation_steps=4
# max_unlearn_steps_list=(284) # 4 times: mmlu/gtbg-64: 572, mmlu/gtbg-256: 284
# random_seeds=(42) 
# max_len=512

# hubble-500b-keep
# forget_topics=(hubble_mmlu-train-unlearn-64)
# retain_topic="hubble_mmlu-train-keep-64"
# bash_key="hubble-500b-keep"
# wandb_project=""
# alphas=(100 1000 10000)
# steering_coeffs_list=(5 50 500)
# lrs=(5e-5 1e-5)
# batch_size=1
# gradient_accumulation_steps=4
# max_unlearn_steps_list=(572) # 4 times: mmlu/gtbg-64: 572, mmlu/gtbg-256: 284
# random_seeds=(42) 
# max_len=512


# hubble-500b-wikitext
# forget_topics=(hubble_mmlu-train-unlearn-256)
# retain_topic="wikitext-train"
# bash_key="hubble-500b"
# wandb_project=""
# alphas=(100 1000 10000)
# steering_coeffs_list=(5 50 500)
# lrs=(5e-5 1e-5)
# batch_size=1
# gradient_accumulation_steps=4
# max_unlearn_steps_list=(568) # 8 times: mmlu/gtbg-64: 1144, mmlu/gtbg-256: 568
# random_seeds=(42) 
# max_len=512

# hubble-500b-keep
# forget_topics=(hubble_gutenberg-train-unlearn-256)
# retain_topic="hubble_gutenberg-train-keep-256"
# bash_key="hubble-500b-keep"
# wandb_project=""
# alphas=(100 1000 10000)
# steering_coeffs_list=(5 50 500)
# lrs=(5e-5 1e-5)
# batch_size=1
# gradient_accumulation_steps=4
# max_unlearn_steps_list=(568) # 8 times: mmlu/gtbg-64: 1144, mmlu/gtbg-256: 568
# random_seeds=(42) 
# max_len=512

# hubble-500b-wikitext
# forget_topics=(hubble_mmlu-train-unlearn-256)
# retain_topic="wikitext-train"
# bash_key="hubble-500b-keep"
# wandb_project=""
# alphas=(100 1000 10000)
# steering_coeffs_list=(5 50 500)
# lrs=(5e-4 1e-4)
# batch_size=1
# gradient_accumulation_steps=4
# max_unlearn_steps_list=(568) # 8 times: mmlu/gtbg-64: 1144, mmlu/gtbg-256: 568
# random_seeds=(42) 
# max_len=512

# hubble-500b-keep
# forget_topics=(hubble_mmlu-train-unlearn-256)
# retain_topic="hubble_mmlu-train-keep-256"
# bash_key="hubble-500b-keep"
# wandb_project=""
# alphas=(100 1000 10000)
# steering_coeffs_list=(5 50 500)
# lrs=(5e-4 1e-4)
# batch_size=1
# gradient_accumulation_steps=4
# max_unlearn_steps_list=(568) # 8 times: mmlu/gtbg-64: 1144, mmlu/gtbg-256: 568
# random_seeds=(42) 
# max_len=512

for forget_topic in "${forget_topics[@]}"; do
    for alpha in "${alphas[@]}"; do
        for max_unlearn_steps in "${max_unlearn_steps_list[@]}"; do
            for lr in "${lrs[@]}"; do
                for steering_coeff in "${steering_coeffs_list[@]}"; do
                    for random_seed in "${random_seeds[@]}"; do
                        run_name="${unlearn_method}_${forget_topic}_${model_name}_${bash_key}_${alpha}_${random_seed}"
                        model_save_path="/path/to/output/${run_name}"

                        echo "Running unlearn.py with alpha=$alpha, max_unlearn_steps=$max_unlearn_steps, lr=$lr, steering_coeff=$steering_coeff, random_seed=$random_seed"
                        conda activate unlearn

                        output=$(python unlearn.py --unlearn_method $unlearn_method \
                                            --model_name_or_path  $model_name_or_path \
                                            --forget_topic $forget_topic \
                                            --retain_topic $retain_topic \
                                            --bash_key $bash_key \
                                            --model_save_path $model_save_path \
                                            --alpha $alpha \
                                            --max_unlearn_steps $max_unlearn_steps \
                                            --max_len $max_len \
                                            --lr $lr \
                                            --batch_size $batch_size \
                                            --gradient_accumulation_steps $gradient_accumulation_steps \
                                            --steering_coeff $steering_coeff \
                                            --random_seed $random_seed \
                                            --wandb_project $wandb_project 2>&1) 
                        last_line=$(echo "$output" | tail -n 1)

                        echo "Finished running unlearn.py with alpha=$alpha, max_unlearn_steps=$max_unlearn_steps, lr=$lr, steering_coeff=$steering_coeff, random_seed=$random_seed"

                        echo "$output"

                        # Check if the last line starts with "wandb id:"
                        if [[ $last_line == wandb\ id:* ]]; then
                            # Extract the run_id
                            run_id=$(echo $last_line | cut -d' ' -f3)
                            # Run lm-eval with the captured wandb run_id and the specified tasks and batch size
                            echo "Starting lm-eval for wandb run_id: $run_id"
                            lm-eval --model vllm  \
                                    --model_args pretrained=${model_save_path},trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.8,enforce_eager=True  \
                                    --tasks hubble_mmlu,tinyMMLU,tinyWinogrande,tinyHellaswag,wikitext \
                                    --batch_size auto \
                                    --wandb_args project=$wandb_project,id=$run_id,resume=allow

                            # Remove the model directory after successful lm-eval
                            echo "Removing model directory: ${model_save_path}"
                            rm -r "${model_save_path}"
                            
                        else
                            # Skip lm-eval if unlearning (training) failed
                            echo "ERROR: unlearn.py failed, skipping lm-eval."
                        fi
                    done
                done
            done
        done
    done
done