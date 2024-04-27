import json
import random

import torch
from torch.utils.data import Dataset, IterableDataset
# from datasets import Dataset
from datasets.table import Table
from lora_api import IGNORE_INDEX
from metrics import sub_percentage
class MWPDatasetForLLM(Dataset):
    def __init__(self, data_path, data_type, data_args, tokenizer):
        self.data_path = data_path
        self.feature = []
        self.data = {}
        self.data_types = data_type
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.get_dataset()

    def get_dataset(self):
        if self.data_path.endswith("json"):
            with open(self.data_path, 'r', encoding="utf-8") as f:
                lines = json.load(f)
                for item in lines:
                    query, response = item["prompt"], item["answer"]
                    # print(query)
                    # print(response)
                    # assert 0 == 1
                    self.feature.append((query, response))
        else:
            with open(self.data_path, 'r', encoding="utf-8") as f:
                lines = f.readlines()
                for item in lines:
                    query, response = item.strip().split('\t')
                    self.feature.append((query, response))
        self.data = process2batch(self.feature, self.data_types, self.data_args, self.tokenizer)

    def __getitem__(self, item):
        return {"input_ids": self.data["input_ids"][item], "labels": self.data["labels"][item]}

    def __len__(self):
        return len(self.data["input_ids"])


def process2batch(data, type, data_args, tokenizer):

    def preprocess_evaluation_dataset(examples):
        # v1: build inputs with format `X [gMASK] <sop>` and labels with format `Y [gMASK] <sop>`
        # v2: build inputs with format `[gMASK] sop X` and labels with format `[gMASK] sop Y`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in examples:
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            # target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            new_target_ids = []
            new_target_ids.append(30910)  # “_” 可能是一个起始符号
            for s in answer:  # 单独处理算术符号，不然会连在一起
                # if s in "+-*/^=":
                #     new_target_ids.append(tokenizer._convert_token_to_id("#"))
                #     new_target_ids.append(tokenizer._convert_token_to_id(s))
                #     # new_target_ids.append(tokenizer._convert_token_to_id("#"))
                # else:
                new_target_ids.append(tokenizer._convert_token_to_id(s))
            target_ids = new_target_ids
            if len(source_ids) > data_args.max_source_length - 2:  # gmask and sop tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(target_ids) > data_args.max_target_length - 2:  # gmask and sop tokens
                target_ids = target_ids[:data_args.max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids)
            labels = tokenizer.build_inputs_with_special_tokens(target_ids)

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_supervised_dataset(examples):
        # v1: build inputs with format `X [gMASK] <sop> Y <eop>` and labels with format `[IGNORE] ... [IGNORE] Y <eop>`
        # v2: build inputs with format `[gMASK] sop X Y </s>` and labels with format `[IGNORE] ... [IGNORE] Y </s>`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in examples:
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            # print(source_ids[-1])
            # print(tokenizer._convert_token_to_id("</s>"))
            # assert 0 == 1
            # target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            new_target_ids = []
            new_target_ids.append(30910) # “_” 可能是一个起始符号
            for s in answer: # 单独处理算术符号，不然会连在一起
                new_target_ids.append(tokenizer._convert_token_to_id(s))
                # if s in "+-*/^=":
                #     new_target_ids.append(tokenizer._convert_token_to_id("#"))
                #     new_target_ids.append(tokenizer._convert_token_to_id(s))
                    # new_target_ids.append(tokenizer._convert_token_to_id("#"))
                # else:

            # print(new_target_ids)
            # print(tokenizer._convert_token_to_id("[OPS]"))
            # print(tokenizer._convert_token_to_id("[OPE]"))
            # print(target_ids)
            # print(tokenizer.convert_tokens_to_string(new_target_ids))
            # for ids in new_target_ids:
            #     print(tokenizer._convert_id_to_token(ids))
            # for ids in target_ids:
            #     print(tokenizer._convert_id_to_token(ids))
            # assert 0 == 1
            target_ids = new_target_ids

            # instruction = "根据要求解的问题和已知的部分答案，填补缺失的步骤："
            # new_prompt = prompt.replace("回答以下数学单词问题，并给出解析解：", instruction)
            #
            # mask_source_ids = tokenizer.encode(text=new_prompt, add_special_tokens=False)
            #
            # span_input_ids, span_labels = prepross_train_data_sim_pt(mask_source_ids, target_ids, tokenizer, "sub_problem_span", data_args)
            #
            # if span_input_ids is not None:
            #     model_inputs["input_ids"].append(span_input_ids)
            #     model_inputs["labels"].append(span_labels)

            if len(source_ids) > data_args.max_source_length - 2:  # gmask and sop tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(target_ids) > data_args.max_target_length - 1:  # eos token
                target_ids = target_ids[:data_args.max_target_length - 1]

            context_length = len(source_ids) + 2  # gmask and sop tokens
            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
            labels = [IGNORE_INDEX] * context_length + input_ids[context_length:]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

        return model_inputs

    if type == 'train':
        return preprocess_supervised_dataset(data)
    elif type == 'valid':
        return preprocess_evaluation_dataset(data)
    else:
        raise ValueError("dataset type must be train or valid")


def prepross_train_data_sim_pt(source_ids:list, target_ids:list, tokenizer, mask_type:str, data_args:object):
    # mask token or sapn to model joint distribution P(X,Y)
    priority_index = [0,0,1,1,2]
    syms_ids = []
    for s in "+-*/^":
        syms_ids.append(tokenizer._convert_token_to_id(s))
    sym_priority = {}
    for t, item in enumerate(syms_ids):
        sym_priority[item] = priority_index[t]
    left_open_paren = tokenizer._convert_token_to_id("(")
    right_close_paren = tokenizer._convert_token_to_id(")")
    equal_sym = tokenizer._convert_token_to_id("=")
    syms_index_tgt_ = []
    for index, ids in enumerate(target_ids):
        if ids in syms_ids:
            syms_index_tgt_.append(index)
    if len(syms_index_tgt_) != 0:

        # print(len(syms_index_tgt_))
        # print(syms_ids)
        # print(target_ids)
        # print(tokenizer._convert_token_to_id("+"))
        # print(tokenizer._convert_token_to_id("-"))
        # print(tokenizer._convert_token_to_id("*"))
        # print(tokenizer._convert_token_to_id("/"))
        # assert 0 == 1
        random.seed(len(source_ids) + len(target_ids))
        chosen_index = random.randint(0, len(syms_index_tgt_)-1)
        mask_ids = tokenizer._convert_token_to_id("[MASK]")

        start = syms_index_tgt_[chosen_index]
        end = syms_index_tgt_[chosen_index] + 1
        if mask_type == "sub_problem_span":
            # only syms chosen as span when
            # "...) op_sym ..."
            # or "... op_sym (..."
            # or "...) op_sym (... "
            if target_ids[syms_index_tgt_[chosen_index]-1] == right_close_paren or \
                    target_ids[syms_index_tgt_[chosen_index]+1] == left_open_paren:
                start = syms_index_tgt_[chosen_index]
                end = syms_index_tgt_[chosen_index] + 1
            # "A op_sym B" span chosen as span when
            # "...( A op_sym B )..."
            # "... op_sym1 A op_sym B )..."
            # "...( A op_sym B op_sym2 ..."
            # " ...op_sym1 A op_sym B op_sym2 ... "
            elif target_ids[syms_index_tgt_[chosen_index]+1] not in syms_ids and \
                    target_ids[syms_index_tgt_[chosen_index]-1] not in syms_ids:
                start = syms_index_tgt_[chosen_index] - 1
                end = syms_index_tgt_[chosen_index] + 1
                while target_ids[start] not in syms_ids and target_ids[start] != left_open_paren and \
                        target_ids[start] != 30910:
                    start -= 1

                while target_ids[end] not in syms_ids and target_ids[end] != right_close_paren and \
                        target_ids[end] != equal_sym:
                    end += 1

                if target_ids[start] in syms_ids:
                    if sym_priority[target_ids[start]] > sym_priority[target_ids[syms_index_tgt_[chosen_index]]]:
                        start = syms_index_tgt_[chosen_index]
                        end = syms_index_tgt_[chosen_index] + 1
                elif target_ids[end] in syms_ids:
                    if sym_priority[target_ids[end]] > sym_priority[target_ids[syms_index_tgt_[chosen_index]]]:
                        start = syms_index_tgt_[chosen_index]
                        end = syms_index_tgt_[chosen_index] + 1
                else:
                    start += 1

        span = target_ids[start:end]

        new_target_ids = [30910] + span
        new_source_ids = target_ids[:start] + [mask_ids] + target_ids[end:]

        if len(new_source_ids) > data_args.max_source_length - 2:  # gmask and sop tokens
            new_source_ids = new_source_ids[:data_args.max_source_length - 2]

        context_length = len(new_source_ids) + 2  # gmask and sop tokens

        input_ids = tokenizer.build_inputs_with_special_tokens(new_source_ids, new_target_ids)
        labels = [IGNORE_INDEX] * context_length + input_ids[context_length:]

        # try:
        eq_ids = target_ids[:start] + span + target_ids[end:]
        eq = tokenizer.decode(eq_ids)

        s, a = eq.split("=")
        s = sub_percentage(s)
        s = s.replace("^", "**")
        # try:
        #     assert abs(eval(s) - eval(a)) < 1e-4
        # except:
        #     print(1111)
        #     print(eq)
        #     print(1234)
        #
        #     print(f"chosen:{target_ids[syms_index_tgt_[chosen_index]]}")
        #     print(f"start_token:{tokenizer.decode(target_ids[syms_index_tgt_[chosen_index]])}")
        #     print(span)
        #     print(tokenizer.decode(span))
        #     print(tokenizer.decode(input_ids))
        #     print(tokenizer.decode(labels))
        #     print(tokenizer.decode(target_ids))

        return input_ids, labels
    else:
        return None, None













