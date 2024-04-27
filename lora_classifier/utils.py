from typing import Sequence, Union, Tuple, Dict, Optional, List, Any
from dataclasses import dataclass
import numpy as np
import torch
from datasets import load_metric
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class compute_metrics:
    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")

        _, labels = eval_preds

        logits = torch.Tensor(_[0])

        pred = torch.softmax(logits, dim=-1).numpy()
        predictions = np.argmax(pred, axis=-1)
        # print(predictions)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}


@dataclass
class compute_metrics_for_pcls:

    eval_batch_size: int

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        load_accuracy = load_metric("accuracy")
        # load_f1 = load_metric("f1")

        _, labels = eval_preds


        logits = torch.Tensor(_[0])

        pred = torch.softmax(logits, dim=-1).numpy()
        predictions = np.argmax(pred, axis=-1)

        # 增加“2”类别标签，有多少label为0, 就有多少label为2, 且根据每个eval batch中label为0的数目在其后面追加相应的label为2的数目
        labels_with_p = []
        # print(len(predictions))
        # print(len(labels))
        # print(labels)
        if len(predictions) > len(labels):
            bsz = self.eval_batch_size
            num_p = 0
            # print("!!!!!!!!!!!")
            for idx, item in enumerate(labels):
                # print(item)
                labels_with_p.append(item)
                if item == 0:
                    num_p += 1
                if (idx + 1) % bsz == 0:
                    labels_with_p += [2 for _ in range(num_p)]
                    num_p = 0
        else:
            labels_with_p = labels
        labels_all = np.array(labels_with_p)
        # print(len(labels_all))
        accuracy = load_accuracy.compute(predictions=predictions, references=labels_all)["accuracy"]
        # f1 = load_f1.compute(predictions=predictions, references=labels_all)["f1"]
        return {"accuracy": accuracy}


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # print(features)
        # print(batch)
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch



