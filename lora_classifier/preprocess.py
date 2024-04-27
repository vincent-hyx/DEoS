import json

from torch.utils.data import Dataset

class ClassifierDatasetForLLM(Dataset):
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
                    query, response, label = item["question"], item["answer"], item["label"]
                    # print(query)
                    # print(response)
                    # assert 0 == 1
                    self.feature.append((query, response, label))
        self.data = process_ori(self.feature, self.data_types, self.data_args, self.tokenizer)


    def __getitem__(self, item):
        return {"input_ids": self.data["input_ids"][item], "labels": self.data["labels"][item]}

    def __len__(self):
        return len(self.data["input_ids"])


class ProcessClassifierDatasetForLLM(Dataset):
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
                    query, response, label, p = item["question"], item["answer"], item["label"], item["p"]
                    # print(query)
                    # print(response)
                    # assert 0 == 1
                    self.feature.append((query, response, label, p))
        self.data = process(self.feature, self.data_types, self.data_args, self.tokenizer)

    def __getitem__(self, item):
        return {"input_ids": self.data["input_ids"][item], "labels": self.data["labels"][item], "p": self.data["p"][item]}

    def __len__(self):
        return len(self.data["input_ids"])


def process(data, type, data_args, tokenizer):
    model_inputs = {"input_ids": [], "labels": [], "p": []}
    for q, a, l, p in data:
        source_ids = tokenizer.encode(text=q, add_special_tokens=False)

        new_target_ids = []
        new_target_ids.append(30910)  # “_” 可能是一个起始符号
        for s in a:  # 单独处理算术符号，不然会连在一起
            new_target_ids.append(tokenizer._convert_token_to_id(s))

        target_ids = new_target_ids

        if len(source_ids) > data_args.max_source_length - 2:  # gmask and sop tokens
            source_ids = source_ids[:data_args.max_source_length - 2]
        if len(target_ids) > data_args.max_target_length - 1:  # eos token
            target_ids = target_ids[:data_args.max_target_length - 1]

        input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)

        model_inputs["input_ids"].append(input_ids)
        model_inputs["labels"].append(l)
        model_inputs["p"].append(p + len(source_ids) + 1)
    return model_inputs


def process_ori(data, type, data_args, tokenizer):
    model_inputs = {"input_ids": [], "labels": []}
    for q, a, l in data:
        source_ids = tokenizer.encode(text=q, add_special_tokens=False)

        new_target_ids = []
        new_target_ids.append(30910)  # “_” 可能是一个起始符号
        for s in a:  # 单独处理算术符号，不然会连在一起
            new_target_ids.append(tokenizer._convert_token_to_id(s))

        target_ids = new_target_ids

        if len(source_ids) > data_args.max_source_length - 2:  # gmask and sop tokens
            source_ids = source_ids[:data_args.max_source_length - 2]
        if len(target_ids) > data_args.max_target_length - 1:  # eos token
            target_ids = target_ids[:data_args.max_target_length - 1]

        input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)

        model_inputs["input_ids"].append(input_ids)
        model_inputs["labels"].append(l)

    return model_inputs
