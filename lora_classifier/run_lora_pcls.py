import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表
import os
from typing import Optional, List

from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel

from lora.callbacks import LogCallback
from lora.dataarg import DataArguments
from lora.finetunearg import FinetuningArguments
from lora.loggings import get_logger

from lora.modelarg import ModelArguments
from transformers import TrainingArguments, TrainerCallback
from lora_classifier.utils import DataCollatorWithPadding, compute_metrics_for_pcls

from lora.parser import get_train_args
from lora.ploting import plot_loss
from lora_classifier.trainer_classifier import PeftTrainer
from lora_classifier.utils import compute_metrics
from model.modeling_chatglm import ChatGLMForSequenceClassification, ChatGLMForSequenceProcessClassification
from model.tokenization_chatglm import ChatGLMTokenizer
from model.configuration_chatglm import ChatGLMConfig
from preprocess import ClassifierDatasetForLLM, ProcessClassifierDatasetForLLM
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logger = get_logger(__name__)

def run_classifier(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    finetuning_args: FinetuningArguments,
    callbacks: Optional[List[TrainerCallback]] = [LogCallback()]
):

    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.model_name_or_path)
    config = ChatGLMConfig
    config.num_labels = 3
    model = ChatGLMForSequenceProcessClassification.from_pretrained(model_args.model_name_or_path, config)
    # new_tokens = ["[OPS]", "[OPE]"]
    # tokenizer.add_tokens(new_tokens)
    # model.resize_token_embeddings(len(tokenizer))
    # if training_args.do_train:
    #     peft_config = LoraConfig(
    #         task_type=TaskType.SEQ_CLS,
    #         inference_mode=False,
    #         r=finetuning_args.lora_rank,
    #         lora_alpha=finetuning_args.lora_alpha,
    #         lora_dropout=finetuning_args.lora_dropout,
    #         target_modules=finetuning_args.lora_target
    #     )
    #
    #     model = get_peft_model(model, peft_config)
    #     logger.info(f"{model.print_trainable_parameters()}")
    # else:
    #     peftconfig = PeftConfig.from_pretrained(model_args.checkpoint_dir[0])
    # peft_config = LoraConfig(
    #             task_type=TaskType.SEQ_CLS,
    #             inference_mode=False,
    #             r=finetuning_args.lora_rank,
    #             lora_alpha=finetuning_args.lora_alpha,
    #             lora_dropout=finetuning_args.lora_dropout,
    #             target_modules=finetuning_args.lora_target
    #         )
    # model = get_peft_model(model, peft_config)
    model = PeftModel.from_pretrained(model, model_args.checkpoint_dir[0], is_trainable=training_args.do_train)
    # model.load_adapter(model_args.checkpoint_dir[0], adapter_name="1", is_trainable=True)

    logger.info(f"{model.print_trainable_parameters()}")
    logger.info(f"loaded checkpoint from {model_args.checkpoint_dir[0]}")

    # train_data = ProcessClassifierDatasetForLLM(os.path.join(data_args.dataset_dir, "classifier_data_math23k.json"),
    #                                  'train', data_args, tokenizer)
    # valid_data = ProcessClassifierDatasetForLLM(os.path.join(data_args.dataset_dir, "classifier_valid_data_math23k.json"),
    #                                  'valid', data_args, tokenizer)
    train_data = ProcessClassifierDatasetForLLM(os.path.join(data_args.dataset_dir, "classifier_data.json"),
                                                'train', data_args, tokenizer)
    valid_data = ProcessClassifierDatasetForLLM(os.path.join(data_args.dataset_dir, "classifier_valid_data.json"),
        'valid', data_args, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = PeftTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_pcls(training_args.eval_batch_size),
        finetuning_args=finetuning_args
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        test_data = ProcessClassifierDatasetForLLM(os.path.join(data_args.dataset_dir, "classifier_test_data.json"),
                                             'valid', data_args, tokenizer)
        predict_results = trainer.predict(test_data, metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

if __name__ == "__main__":
    model_args, data_args, training_args, finetuning_args, general_args = get_train_args()
    run_classifier(model_args, data_args, training_args, finetuning_args)
