import csv
import json

from data.math23k.expr_process import from_prefix_to_infix
def process2txt_infix(csv_path, data_type):
    csv_reader = csv.reader(open(csv_path))
    lines = []
    cls_data = []
    for idx, row in enumerate(csv_reader):
        cls = {}
        if idx > 0:
            q = row[0]
            nums = list(row[1].split(' '))
            prefix_eq = row[2]
            # print(prefix_eq)
            answer = row[3]
            for t in range(len(nums)):
                prefix_eq = prefix_eq.replace(f"number{t}", nums[t])
                q = q.replace(f"number{t}", nums[t])
            infix_eq = from_prefix_to_infix(prefix_eq)
            # print(infix_eq)
            # print(answer)
            # print(eval(infix_eq))
            answer = eval(infix_eq)
            infix_eq = infix_eq + "=" + f"{answer}"
            lines.append(q.strip() + "\t" + infix_eq + "\n")
            cls["question"] = q.strip()
            cls["answer"] = infix_eq
            cls["id"] = idx
            cls["label"] = 1
            cls_data.append(cls)
    return lines, cls_data

if __name__ == "__main__":
    for i in range(5):
        csv_path_train = f'./fold{i}/train.csv'
        csv_path_dev = f'./fold{i}/dev.csv'
        train_lines, train_cls = process2txt_infix(csv_path_train, "train")
        dev_lines, dev_cls = process2txt_infix(csv_path_dev, "dev")
        txt_path_train = f'./fold{i}/train.txt'
        txt_path_dev = f'./fold{i}/dev.txt'
        with open(txt_path_train, 'w', encoding="utf-8") as fw:
            fw.writelines(train_lines)
        with open(txt_path_dev, 'w', encoding="utf-8") as fw:
            fw.writelines(dev_lines)
        txt_path_train = f'./fold{i}/train.json'
        txt_path_dev = f'./fold{i}/dev.json'
        with open(txt_path_train, 'w', encoding="utf-8") as fw:
            json_data = json.dumps(train_cls, indent=4, ensure_ascii=False)
            fw.write(json_data)
        with open(txt_path_dev, 'w', encoding="utf-8") as fw:
            json_data = json.dumps(dev_cls, indent=4, ensure_ascii=False)
            fw.write(json_data)



