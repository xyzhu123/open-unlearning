#!/bin/bash

wandb_project=""

model_infos=(
    "hubble-8b-500b_toks-perturbed-hf /path/to/model"
)
output_dir="/path/to/output"


trainers_experiments=(
    "SatImp mmlu_wikitext open-unlearning/configs/experiment/unlearn/hubble/mmlu_wikitext.yaml"
)
forget_retain_splits=(
    "train train"
)

per_device_train_batch_size=1
gradient_accumulation_steps=16


lrs=(1e-5 5e-5 1e-4)
alphas=(1.0 0.1 0.01)
betas=(5.0 6.0)
beta2=1.0

num_train_epochs_list=(4 8)

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for model_info in "${model_infos[@]}"; do
        model=$(echo $model_info | cut -d' ' -f1)
        model_path=$(echo $model_info | cut -d' ' -f2)
        for trainer_experiment in "${trainers_experiments[@]}"; do
            read -r trainer experiment_name experiment <<< "$trainer_experiment"
            for lr in "${lrs[@]}"; do 
                for num_train_epochs in "${num_train_epochs_list[@]}"; do
                    for beta1 in "${betas[@]}"; do 
                        for alpha in "${alphas[@]}"; do          
                            task_name=${experiment_name}_${model}_${forget_split}_${trainer}_lr${lr}_beta1${beta1}_beta2${beta2}_alpha${alpha}
                            full_output_dir=${output_dir}_${task_name}
                            echo ${task_name}: Unlearning ${model_path} using ${trainer}

                            # Unlearn
                            # change max_steps to num_train_epochs later
                            CUDA_VISIBLE_DEVICES=0 \
                            python src/train.py --config-name=unlearn.yaml \
                            experiment=${experiment} \
                            trainer=${trainer} \
                            task_name=${task_name} \
                            model=${model} \
                            forget_split=${forget_split} \
                            retain_split=${retain_split} \
                            model.model_args.pretrained_model_name_or_path=${model_path} \
                            retain_logs_path=saves/eval/hubble_${model}_${retain_split}/hubble_EVAL.json \
                            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                            trainer.args.do_train=True \
                            trainer.args.do_eval=False \
                            trainer.args.eval_strategy=no \
                            trainer.args.eval_on_start=False \
                            trainer.args.output_dir=$full_output_dir \
                            trainer.args.num_train_epochs=$num_train_epochs \
                            trainer.args.learning_rate=$lr \
                            trainer.method_args.beta1=$beta1 \
                            trainer.method_args.beta2=$beta2 \
                            trainer.method_args.alpha=$alpha \

                            echo "Starting lm-eval with wandb"

                            conda activate hubble_eval

                            lm-eval --model vllm \
                                --model_args pretrained=${full_output_dir},trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.8,enforce_eager=True \
                                --tasks hubble_mmlu,tinyMMLU,tinyWinogrande,tinyHellaswag,wikitext \
                                --batch_size auto \
                                --wandb_args project=$wandb_project
                            
                            echo "Removing model directory: ${full_output_dir}"
                            rm -r "${full_output_dir}"


                        done
                    done
                done
            done
        done
    done
done