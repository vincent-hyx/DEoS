import re

import numpy as np
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union
from transformers.tokenization_utils import PreTrainedTokenizer

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from lora.lora_api import IGNORE_INDEX

OP_LIST = ["+", "-", "*", "/", "^"]
ORDER_DICT = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}

def from_prefix_to_infix(expr:str):
    expression = list(expr.split(" "))
    sym_set = ["+", '-', '*', '/', '^']
    stack = []
    for e in expression:
        stack.append(e)
        if e not in sym_set:
            while len(stack) > 1 and stack[-2] not in sym_set:
                op_b = stack.pop()
                op_a = stack.pop()
                op = stack.pop()
                if len(stack) != 0:
                    res = f"({op_a}{op}{op_b})"
                else:
                    res = f"{op_a}{op}{op_b}"
                stack.append(res)
    while len(stack) > 1:
        op_b = stack.pop()
        op_a = stack.pop()
        op = stack.pop()
        if len(stack) != 0:
            res = f"({op_a}{op}{op_b})"
            print(res)
        else:
            res = f"{op_a}{op}{op_b}"
        stack.append(res)
    assert len(stack) == 1
    return stack[0]

def from_postfix_to_infix(postfix):
    if isinstance(postfix, str):
        postfix = postfix.split(' ')
    stack = []
    for elem in postfix:
        if elem in OP_LIST:
            a, od_a = stack.pop()
            b, od_b = stack.pop()
            od_c = ORDER_DICT[elem]
            if od_a <= od_c:
                a = "( " + a + " )"
            if od_b < od_c:
                b = "( " + b + " )"
            tmp = b + " " + elem + " " + a
            stack.append((tmp, od_c))
        else:
            stack.append((elem, 3))
    assert len(stack) == 1
    return stack[-1][0]


def sub_percentage(new_infix): # 先把 '小数%' 替换，再替换"整数%"， 且每次替换只替换一次
    all_match1 = re.findall(r'\d+\.\d+%', new_infix)  # 匹配类似 1.5%
    if len(all_match1) != 0:
        for item in all_match1:
            # print(data_type)
            # print(infix_target)
            # print(item)
            new_sub_str = f"({item[:-1]}/100)"
            # print(new_sub_str)
            new_infix = new_infix.replace(item, new_sub_str, 1)
            # print(new_infix)
            # assert 0 == 1

    all_match2 = re.findall(r'\d+%', new_infix)  # 匹配类似 5%
    if len(all_match2) != 0:
        for item in all_match2:
            # print(data_type)
            # print(infix_target)
            # print(item)
            new_sub_str = f"({item[:-1]}/100)"
            # print(new_sub_str)
            new_infix = new_infix.replace(item, new_sub_str, 1)
            # print(new_infix)
            # assert 0 == 1
    return new_infix

@dataclass
class ComputeMetrics_MWPacc:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqTrainerForChatGLM.
    """

    tokenizer: PreTrainedTokenizer
    answer_type: str
    data_type: str

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"accuracy": [], "rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        # score_dict = {"value accuracy": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]
                # aaa = rouge.convert_and_evaluate()

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            # score_dict["accuracy"].append(float(len(label) != 0 and pred[:len(label)] == label))

            # edited by xu, modified MWP accuracy

            if self.data_type == "sub":
                _, label = label.split(">>>")
                _, label_eq, labels_ans = label.split('=')
            elif self.data_type == "ori":
                label_eq, labels_ans = label.split('=')

            else:
                raise TypeError("data type must be 'ori' or 'sub'!")
            print("label:" + label)
            print("_________________")
            try:
                if self.data_type == "sub":
                    _, pred = pred.split(">>>")
                    _, pred_eq, pred_ans = pred.split('=')
                elif self.data_type == "ori":
                    pred_eq, pred_ans = pred.split('=')
                else:
                    raise TypeError("data type must be 'ori' or 'sub'!")

                if self.answer_type == "prefix":
                    pred_eq = from_prefix_to_infix(pred_eq)
                if self.answer_type == "postfix":
                    pred_eq = from_postfix_to_infix(pred_eq)

                # 移除“#”号
                pred_eq = pred_eq.replace("#", "")
                # 替换"^"为"**", 以及替换"%"
                pred_eq = pred_eq.replace("^", "**")
                pred_eq = sub_percentage(pred_eq)

                ans = eval(pred_eq)
                if abs(ans - eval(labels_ans)) < 1e-4:
                    score_dict["accuracy"].append(1)
                    print("Right Expression:" + pred)
                    print("-----------------")
                else:
                    score_dict["accuracy"].append(0)
                    print("Wrong Expression:" + pred)
                    print("-----------------")

            except:
                # eval() function not support "%"计算，if "%" leads to exception, we just consider pred_ans.
                if abs(eval(pred_ans) - eval(labels_ans)) < 1e-4:
                    score_dict["accuracy"].append(1)
                    print("Right Answer but Expression can't be calculated:" + pred)
                    print("-----------------")
                else:
                    score_dict["accuracy"].append(0)
                    print("Wrong Answer and Expression can't be calculated:" + pred)
                    print("-----------------")
                # score_dict["accuracy"].append(0)
            finally:
                continue


        return {k: float(np.mean(v)) for k, v in score_dict.items()}