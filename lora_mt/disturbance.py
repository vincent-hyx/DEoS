import json
import os
import random
import sys
from time import sleep

from tqdm import tqdm
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表


from data.math23k.exp_tree import corrupt_expression
from lora.metrics import sub_percentage

def readfile(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def map_num(expr): # there is no "=" in expr
    start = 0
    end = 0
    num_sym = [f"{_}" for _ in range(10)]
    num_sym.append('.')
    num_sym.append("%")
    num_map = []
    new_expr = ''
    times = 0
    while start < len(expr):
        if expr[start] in num_sym:
            end = start
            end += 1
            while expr[end] in num_sym:
                end += 1
            num_map.append(expr[start:end])
            start = end
            new_expr += f'#{times}'
            times += 1
        else:
            new_expr += expr[start]
            start += 1
    return new_expr, num_map

def split_expr(expr):
    start = 0
    end = 0
    num_sym = [f"{_}" for _ in range(10)]
    num_sym.append('.')
    num_sym.append("%")
    new_expr = []
    while start < len(expr):
        if expr[start] in num_sym:
            end = start
            end += 1
            while end < len(expr) and expr[end] in num_sym:
                end += 1
            new_expr.append(f'{expr[start:end]}')
            start = end
        else:
            new_expr.append(expr[start])
            start += 1
    return new_expr

def max_prefix_match(ori_expr, corrupt_expr):
    ori_len = len(ori_expr)
    corrupt_len = len(corrupt_expr)
    length = min(ori_len, corrupt_len)
    score = 0
    p = -2
    i = 0
    while i < length:
        if ori_expr[i] == corrupt_expr[i]:
            score += 1
            p = i
            i += 1
        else:
            break
    return score, p


def select_corrupt_expr(ori_expr): # expr = "1.2 + 3.5 - 40%"
    final_expr = ''
    tmp_score = -1
    final_p = 0
    # print(ori_expr)
    for i in range(10): # math23k: 20
        random.seed(42+i)

        corrupt_expr = corrupt_expression(ori_expr).replace(" ", '')
        # print(corrupt_expr)
        expr = ori_expr.replace(" ", '')

        score, p = max_prefix_match(expr, corrupt_expr)
        if p == -2:
            continue
        # print(corrupt_expr[p])

        tmp_ori = sub_percentage(expr)
        tmp_corrupt = sub_percentage(corrupt_expr)
        tmp_ori = tmp_ori.replace("^", "**")
        tmp_corrupt = tmp_corrupt.replace("^", "**")

        # corrupt_answer = eval(tmp_ori)
        # print(tmp_ori)
        # print(tmp_corrupt)
        # print(eval(tmp_corrupt))
        # assert 0 == 1
        try:
            a = abs(eval(tmp_ori) - eval(tmp_corrupt))
            # corrupt_answer = eval(tmp_corrupt)
        except:
            continue
        if a > 1e-4:
            if score > tmp_score:
                tmp_score = score
                final_expr = corrupt_expr
                final_p = p
            elif score == tmp_score:
                if len(corrupt_expr) < len(corrupt_expr):
                    tmp_score = score
                    final_expr = corrupt_expr
                    final_p = p
            else:
                continue

    try:
        e = sub_percentage(final_expr)
        e = e.replace("^", "**")
        corrupt_answer = eval(e)
    except:
        corrupt_answer = None

    return final_expr, final_p, tmp_score, corrupt_answer


def offline(ori_train_data, tgt_path):
    data = readfile(ori_train_data)
    new_data = []
    for item in tqdm(data, desc="performing offline update"):
        if item["answer"] == "80千米/小时=80.0":
            continue
        cls = {}
        expr, answer = item["answer"].split("=")
        cls["question"] = item["question"]
        cls["label"] = 0
        cls["id"] = "offline"

        if expr == "(((-2.0)*(-15.0))*4.0)*(-1.0)" or expr == "((-2.0*-15.0)*4.0)*-1.0":
            continue

        expr = " ".join(split_expr(expr))
        # print(expr)
        # assert 0 == 1
        final_expr, final_p, score, a = select_corrupt_expr(expr)
        if final_expr == "":
            print("fail to generate")
            continue
        # print(expr)
        # print(final_expr)
        # print(final_p)
        # print(eval(final_expr))
        # sleep(1)
        # print(f"{final_expr[final_p]}")

        assert final_p == score - 1
        cls["p"] = final_p
        if a is not None:
            if a % 1 == 0:
                try:
                    cls["answer"] = final_expr + '=' + f"{a}"
                except:
                    print("large value")
                    cls["answer"] = final_expr + '=' + answer
            else:
                cls["answer"] = final_expr + '=' + f"{a:.4f}"
        else:
            # print("can't be calculated")
            cls["answer"] = final_expr + '=' + answer
        new_data.append(cls)
    with open(tgt_path, "w", encoding="utf-8") as fw:
        json_data = json.dumps(new_data, indent=4, ensure_ascii=False)
        fw.write(json_data)


if __name__ == "__main__":
    # dataset = "math23k"
    # dataset = "mawps-single-five-fold"
    dataset = "mawps"
    if dataset == "math23k":
        tgt_path = f"./lora_mt/cls_data/classifier_data_{dataset}_offline.json"
        ori_path = f"./data/{dataset}/classifier_train_data.json"
        offline(ori_path, tgt_path)
        tgt_path = f"./lora_mt/cls_data/classifier_valid_data_{dataset}_offline.json"
        ori_path = f"./data/{dataset}/classifier_valid_data.json"
        offline(ori_path, tgt_path)
    elif dataset == "mawps-single-five-fold":
        for i in range(5):
            data_kw = f"fold{i}"
            os.makedirs(f'./lora_mt/cls_data/{dataset}/{data_kw}/', exist_ok=True)
            tgt_path = f"./lora_mt/cls_data/{dataset}/{data_kw}/classifier_data_offline.json"
            ori_path = f"./data/{dataset}/{data_kw}/cls_train.json"
            offline(ori_path, tgt_path)
            tgt_path = f"./lora_mt/cls_data/{dataset}/{data_kw}/classifier_valid_data_offline.json"
            ori_path = f"./data/{dataset}/{data_kw}/cls_test.json"
            offline(ori_path, tgt_path)
    elif dataset == "mawps":
        for i in range(5):
            data_kw = f"fold{i}"
            os.makedirs(f'./lora_mt/cls_data/{dataset}/{data_kw}/', exist_ok=True)
            tgt_path = f"./lora_mt/cls_data/{dataset}/{data_kw}/classifier_data_offline.json"
            ori_path = f"./data/cv_{dataset}/{data_kw}/train.json"
            offline(ori_path, tgt_path)
            tgt_path = f"./lora_mt/cls_data/{dataset}/{data_kw}/classifier_valid_data_offline.json"
            ori_path = f"./data/cv_{dataset}/{data_kw}/dev.json"
            offline(ori_path, tgt_path)
    else:
        print(f"only dataset name: math23k or mawps-single-five-fold or mawps!")
        print("check your dataset name!")





