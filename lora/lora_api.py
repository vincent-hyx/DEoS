from transformers import AutoTokenizer, AutoModel, PreTrainedModel, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from typing import Optional
import torch
import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, TRAINING_ARGS_NAME
from transformers.modeling_utils import unwrap_model, load_sharded_checkpoint
from transformers.trainer import PredictionOutput
from lora.finetunearg import FinetuningArguments
from lora.loggings import get_logger
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表
from model.tokenization_chatglm import ChatGLMTokenizer
from model.modeling_chatglm import ChatGLMForConditionalGeneration

IGNORE_INDEX = -100

VALUE_HEAD_FILE_NAME = "value_head.bin"

FINETUNING_ARGS_NAME = "finetuning_args.json"

LAYERNORM_NAMES = ["layernorm"]

METHODS = ["full", "freeze", "p_tuning", "lora"]

SUPPORTED_MODELS = {
    "ChatGLM-6B": "THUDM/chatglm-6b",
    "ChatGLM2-6B": "THUDM/chatglm2-6b"
}


logger = get_logger(__name__)




def get_state_dict(model: torch.nn.Module, trainable_only: Optional[bool] = True) -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in model.named_parameters():
        if (not trainable_only) or v.requires_grad:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

    return filtered_state_dict


def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    weights_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if os.path.exists(weights_file):
        model_state_dict = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False) # skip missing keys
    elif os.path.exists(os.path.join(checkpoint_dir, WEIGHTS_INDEX_NAME)):
        load_sharded_checkpoint(model, checkpoint_dir, strict=False)
    else:
        logger.warning("Provided path ({}) does not contain pre-trained weights.".format(checkpoint_dir))
        return False
    return True


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    if not os.path.exists(valuehead_file):
        logger.warning("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
    return True

class PeftTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self._remove_log()

    def _remove_log(self):
        if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = unwrap_model(self.model)

        if hasattr(model, "pretrained_model"): # for models with valuehead (currently using LoRA only)
            backbone_model = getattr(model, "pretrained_model")
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
        else:
            backbone_model = model

        if isinstance(backbone_model, PeftModel): # LoRA tuning
            backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))
        elif isinstance(backbone_model, PreTrainedModel): # freeze/full-tuning or p_tuning
            backbone_model.config.use_cache = True
            backbone_model.save_pretrained(
                output_dir,
                state_dict=get_state_dict(backbone_model, trainable_only=(self.finetuning_args.finetuning_type != "full")),
                safe_serialization=self.args.save_safetensors
            )
            backbone_model.config.use_cache = False
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
        else:
            logger.warning("No model to save.")

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string() + "\n")
        self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

    def _load_best_model(self):
        r"""
        Loads trainable parameters from model checkpoint.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")

        model = unwrap_model(self.model)
        backbone_model = getattr(model, "pretrained_model") if hasattr(model, "pretrained_model") else model

        if isinstance(backbone_model, PeftModel):
            backbone_model.load_adapter(self.state.best_model_checkpoint, backbone_model.active_adapter)
            if hasattr(model, "v_head") and load_valuehead_params(model, self.state.best_model_checkpoint):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })
        else: # freeze/full-tuning or p_tuning
            load_trainable_params(backbone_model, self.state.best_model_checkpoint)

class Seq2SeqTrainerForChatGLM(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        input_ids = inputs["input_ids"]
        if self.args.do_train:
            self.args.predict_with_generate = True
        # loss, generated_tokens, labels = super().prediction_step(
        #     model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        # )
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if self.args.do_train:
            self.args.predict_with_generate = False
        # print(generated_tokens)
        # print(input_ids.shape)
        # tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
        # print(tokenizer.batch_decode(input_ids))
        # print(type(generated_tokens))
        # print(f"len:{len(generated_tokens)}")
        # print(input_ids.size(-1))
        # print(generated_tokens[0].shape)
        # # print(generated_tokens[1])
        # print(len(generated_tokens[1]))
        # print(type(generated_tokens[1]))
        # print(type(generated_tokens[1][0]))
        # print(len(generated_tokens[1][0]))
        # print(generated_tokens[1][2][0].shape)
        # print(generated_tokens[1][2][1].shape)
        # print(generated_tokens[1][2][0])
        # print(tokenizer.batch_decode(labels))
        # print(labels.shape)
        generated_tokens = generated_tokens[:, input_ids.size(-1):] if generated_tokens is not None else None
        return (loss, generated_tokens, labels)

    def save_predictions(
        self,
        predict_results: PredictionOutput
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.json")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[dict] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append({"label": label, "predict": pred})
            json_data = json.dumps(res, indent=4, ensure_ascii=False)
            writer.write(json_data)

    def prediction_step_with_attention(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()

        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()

        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        generated_tokens = self.model.generate(**inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                    if self.args.do_predict:
                        print(outputs)
                        attn = outputs["attentions"]
                        print(attn.shape)
                        assert 0 == 1
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels



def load_model_tokenizer_lora(model_args, finetune_args, do_train):
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.model_name_or_path)
    model = ChatGLMForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    print(model_args.checkpoint_dir[0])
    if model_args.checkpoint_dir[0] == "first_generate":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=finetune_args.lora_alpha,
            lora_dropout=finetune_args.lora_dropout,
            target_modules=finetune_args.lora_target
        )

        model = get_peft_model(model, peft_config)
        logger.info(f"{model.print_trainable_parameters()}")
    else:
        peft_config = PeftConfig.from_pretrained(model_args.checkpoint_dir[0])
        model = PeftModel.from_pretrained(model, model_args.checkpoint_dir[0], is_trainable=do_train)
        logger.info(f"{model.print_trainable_parameters()}")
        logger.info(f"loaded checkpoint from {model_args.checkpoint_dir[0]}")
    return model, tokenizer






# if __name__ == '__main__':
#     # lora_train("../model/")