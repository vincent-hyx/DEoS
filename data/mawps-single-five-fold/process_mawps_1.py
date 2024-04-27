import csv
import json
import os

from data.math23k.expr_process import from_prefix_to_infix
def process2txt_infix(json_path, data_type):
    json_data = json.load(open(json_path))
    lines = []
    cls_data = []
    for idx, item in enumerate(json_data):
        cls = {}
        q = item["sQuestion"]
        eq = item["new_equation"]
        eq = eq.replace(" ", "")
        eq = eq[2:]
        answer = item["answer"]
        infix_eq = eq + "=" + f"{answer}"
        lines.append(q.strip() + "\t" + infix_eq + "\n")
        cls["question"] = q.strip()
        cls["answer"] = infix_eq
        cls["id"] = idx
        cls["label"] = 1
        cls_data.append(cls)
    return lines, cls_data

if __name__ == "__main__":
    for i in range(5):
        csv_path_train = f'./train_{i}.json'
        csv_path_dev = f'./test_{i}.json'
        train_lines, train_cls = process2txt_infix(csv_path_train, "train")
        dev_lines, dev_cls = process2txt_infix(csv_path_dev, "dev")
        txt_path_train = f'./fold{i}/train.txt'
        txt_path_dev = f'./fold{i}/test.txt'
        os.makedirs(f'./fold{i}/', exist_ok=True)
        with open(txt_path_train, 'w', encoding="utf-8") as fw:
            fw.writelines(train_lines)
        with open(txt_path_dev, 'w', encoding="utf-8") as fw:
            fw.writelines(dev_lines)
        txt_path_train = f'./fold{i}/cls_train.json'
        txt_path_dev = f'./fold{i}/cls_test.json'
        with open(txt_path_train, 'w', encoding="utf-8") as fw:
            json_data = json.dumps(train_cls, indent=4, ensure_ascii=False)
            fw.write(json_data)
        with open(txt_path_dev, 'w', encoding="utf-8") as fw:
            json_data = json.dumps(dev_cls, indent=4, ensure_ascii=False)
            fw.write(json_data)



