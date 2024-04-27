import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表

import json
import os
from typing import Optional, List

from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments
from transformers import TrainerCallback
from finetunearg import FinetuningArguments
from modelarg import ModelArguments
from dataarg import DataArguments
from callbacks import LogCallback
from parser import get_train_args
from dataprocess import MWPDatasetForLLM, process2batch
from lora_api import load_model_tokenizer_lora, Seq2SeqTrainerForChatGLM
from collator import DataCollatorForChatGLM
from metrics import ComputeMetrics_MWPacc
from misc import get_logits_processor
from ploting import plot_loss
from torch.utils.data import ConcatDataset


def run_sft(
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: Seq2SeqTrainingArguments,
        finetuning_args: FinetuningArguments,
        callbacks: Optional[List[TrainerCallback]] = [LogCallback()]
):
    model, tokenizer = load_model_tokenizer_lora(model_args, finetuning_args, training_args.do_train)
    # train_dataset = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "answer_math23k_processed_train.txt"),
    #                                  'train', data_args, tokenizer)
    # # train_dataset1 = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "answer_math23k_processed_train.txt"),
    # #                                  'train', data_args, tokenizer)
    # # concatdataset_train = ConcatDataset([train_dataset, train_dataset1])
    # valid_dataset = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "answer_math23k_processed_valid.txt"),
    #                                  'valid', data_args, tokenizer)
    # predict_dataset = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "answer_math23k_processed_train.txt"),
    #                                    'valid', data_args, tokenizer)

    train_dataset = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "train.txt"),
                                     'train', data_args, tokenizer)
    # train_dataset1 = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "answer_math23k_processed_train.txt"),
    #                                  'train', data_args, tokenizer)
    # concatdataset_train = ConcatDataset([train_dataset, train_dataset1])
    valid_dataset = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "dev.txt"), # mawps dev.txt mawps-single-five-fold test.txt
                                     'valid', data_args, tokenizer)
    predict_dataset = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "train.txt"),
                                       'valid', data_args, tokenizer)
    # valid_dataset = MWPDatasetForLLM(os.path.join(data_args.dataset_dir, "icl_test.json"),'valid', data_args, tokenizer)
    # print(train_dataset[0])

    # train_dataset = preprocess_supervised_dataset(train_dataset, tokenizer, data_args)
    # valid_dataset = preprocess_evaluation_dataset(valid_dataset, tokenizer, data_args)
    # train_dataset = process2batch(train_dataset, 'train', data_args, tokenizer)
    # valid_dataset = process2batch(valid_dataset, 'valid', data_args, tokenizer)
    # new_tokens = ["[OPS], [OPE]"]
    # tokenizer.add_tokens(new_tokens)
    # model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForChatGLM(tokenizer=tokenizer,
                                           model=model,
                                           ignore_pad_token_for_loss=(
                                                       data_args.ignore_pad_token_for_loss and not training_args.predict_with_generate))

    # fixing decoder parameters
    training_args.generation_max_length = training_args.generation_max_length if \
        training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.eval_num_beams if \
        data_args.eval_num_beams is not None else training_args.generation_num_beams

    # Initialize our Trainer
    trainer = Seq2SeqTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics_MWPacc(tokenizer, "infix", "ori"),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )

    # Keyword arguments for `model.generate`
    if training_args.do_train and training_args.do_eval:
        gen_kwargs = {
            "do_sample": True,
            "top_p": 1,  # 0.75
            "max_new_tokens": data_args.max_target_length + 1,
            "temperature": 1,  # 0.95
            "num_beams": 5,
            "logits_processor": get_logits_processor()
        }
    else:
        gen_kwargs = {
            "do_sample": True,
            "top_p": 1,  # 0.75
            "max_new_tokens": data_args.max_target_length + 1,
            "temperature": 1,  # 0.95
            "num_beams": 5,
            "logits_processor": get_logits_processor()
        }

    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
        return train_result

        # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return metrics

        # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
        return predict_results


if __name__ == '__main__':
    model_args, data_args, training_args, finetuning_args, general_args = get_train_args()
    # print(model_args)
    # print(data_args)
    # print(training_args)
    # print(finetuning_args)
    # print(general_args)
    if general_args.stage == "sft":
        if training_args.do_train is False:
            checkpoint = model_args.checkpoint_dir[0]
            print(checkpoint)
            checkpoint_list = list(checkpoint.split(" "))
            print(checkpoint_list)
            result_acc = {}
            best_acc = 0
            best_epoch = ""
            if len(checkpoint_list) == 1:
                checkpoint_dir = os.path.join(checkpoint_list[0])
                model_args.checkpoint_dir = [checkpoint_dir]
                _ = run_sft(model_args, data_args, training_args, finetuning_args)
            else:
                for i in tqdm(range(1, len(checkpoint_list))):
                    checkpoint_dir = os.path.join(checkpoint_list[0], checkpoint_list[i])
                    # print(checkpoint_dir)
                    model_args.checkpoint_dir = [checkpoint_dir]
                    metric = run_sft(model_args, data_args, training_args, finetuning_args)
                    result_acc[checkpoint_list[i]] = metric["eval_accuracy"]
                    if metric["eval_accuracy"] >= best_acc:
                        best_acc = metric["eval_accuracy"]
                        best_epoch = checkpoint_list[i]
                path_eval_acc = os.path.join(checkpoint_list[0], "result_acc.txt")

                with open(path_eval_acc, 'w') as f:
                    f.write(json.dumps(result_acc, indent=2))
                    f.write(f"\nbest_epoch:{best_epoch}||best_acc:{best_acc}")
        else:
            _ = run_sft(model_args, data_args, training_args, finetuning_args)
    # elif general_args.stage == "rm":
    #     run_rm(model_args, data_args, training_args, finetuning_args)
    # elif general_args.stage == "ppo":
    #     run_ppo(model_args, data_args, training_args, finetuning_args)
