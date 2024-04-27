import random
import sys
from pathlib import Path

from tqdm import tqdm



sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表
import json
from lora.metrics import sub_percentage
from lora_mt.disturbance import max_prefix_match, split_expr, select_corrupt_expr


def check(pred, label, question):
    new_cls_data = []
    for p, l, q in zip(pred, label, question):
        cls = {}
        cls["question"] = q
        cls["id"] = "online"
        try:
            p_eq, pseudo_ans = p.split("=")
            l_eq, l_ans = l.split("=")

            cls["answer"] = p_eq + "="

            p_eq = p_eq.replace("#", "")
            # 替换"^"为"**", 以及替换"%"
            p_eq = p_eq.replace("^", "**")
            p_eq = sub_percentage(p_eq)

            p_ans = eval(p_eq)
            if abs(p_ans - eval(l_ans)) < 1e-4:
                cls["answer"] += l_ans
                cls["label"] = 1
            else:
                cls["answer"] += p_ans
                cls["label"] = 0
        except:
            cls["answer"] = p
            cls["label"] = 0
        finally:
            new_cls_data.append(cls)
            continue
    return new_cls_data


def readfile(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def online_update(tgt_path, ori_path, pred_res_path):
    train_data = readfile(ori_path)
    pred = []
    label = []
    question = []
    for item in tqdm(train_data, desc="performing online update"):
        label.append(item["answer"])
        question.append(item["question"])
    pred_data = readfile(pred_res_path)
    for item in pred_data:
        pred.append(item["predict"])
    new_cls_data = check(pred, label, question)
    with open(tgt_path, "w", encoding="utf-8") as fw:
        json_data = json.dumps(new_cls_data, indent=4, ensure_ascii=False)
        fw.write(json_data)


# To process online data that answer is wrong to obtain p value
# meanwhile remove completely same data from online and original data
def process_online_data(online_path, ori_path, mode="mix"):
    new_online_data = []
    online_data = readfile(online_path)
    ori_data = readfile(ori_path)
    if mode == "mix":
        for online, ori in zip(online_data, ori_data):
            if online["label"] == 0:
                score, p = max_prefix_match(ori["answer"], online["answer"])
                online["p"] = p
                new_online_data.append(online)
            else:
                if online["answer"] != ori["answer"]:
                    online["p"] = -2
                    new_online_data.append(online)

                    cls = {}
                    expr, answer = online["answer"].split("=")
                    cls["question"] = online["question"]
                    cls["label"] = 0
                    cls["id"] = "online_offline"

                    expr = " ".join(split_expr(expr))
                    # print(expr)
                    # assert 0 == 1
                    final_expr, final_p, score, a = select_corrupt_expr(expr)
                    if final_expr == "":
                        print("fail to generate")
                        continue

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
                    new_online_data.append(cls)
            ori["p"] = -2
            new_online_data.append(ori)
    else: # remove consistency solutions
        for online, ori in zip(online_data, ori_data):
            if online["label"] == 1:
                if online["answer"] != ori["answer"]:
                    new_online_data.append(ori)
                else:
                    online["p"] = -2
                    new_online_data.append(online)
            else:
                online["p"] = -2
                new_online_data.append(online)

    with open(online_path, "w", encoding="utf-8") as fw:
        json_data = json.dumps(new_online_data, indent=4, ensure_ascii=False)
        fw.write(json_data)






# def offline_update(con_sample_path, merge_path, merge=True):
#     data = []
#     with open(con_sample_path, "r", encoding="utf-8") as fr:
#         lines = fr.readlines()
#         for li in lines:
#             cls = {}
#             try:
#                 q, a, label = li.strip().split("\t")
#             except:
#                 continue
#             cls["question"] = q
#             cls["answer"] = a
#             cls["label"] = eval(label)
#             cls["id"] = "offline"
#             data.append(cls)
#     old_data = readfile(merge_path)
#     new_data = old_data + data
#     if merge:
#         with open(merge_path, "w", encoding="utf-8") as fw:
#             json_data = json.dumps(new_data, indent=4, ensure_ascii=False)
#             fw.write(json_data)
#     else:
#         with open(merge_path, "w", encoding="utf-8") as fw:
#             json_data = json.dumps(data, indent=4, ensure_ascii=False)
#             fw.write(json_data)


def extend_dataset(path1, path2, tgt_path,  mode="only online", rondom_sample=False):
    ori_data = readfile(path1)
    for item in ori_data:
        if "p" not in item:
            item["p"] = -2
    cs_data = readfile(path2)
    print(f"mode: {mode}")
    if mode == "mix":
        if rondom_sample:
            random.seed(2024)
            ori_data = random.sample(ori_data, 800)
            cs_data = random.sample(cs_data, 800)
        new_data = ori_data + cs_data
        with open(tgt_path, "w", encoding="utf-8") as fw:
            json_data = json.dumps(new_data, indent=4, ensure_ascii=False)
            fw.write(json_data)
    elif mode == "only offline":
        with open(tgt_path, "w", encoding="utf-8") as fw:
            json_data = json.dumps(cs_data, indent=4, ensure_ascii=False)
            fw.write(json_data)
    elif mode == "only online":
        with open(tgt_path, "w", encoding="utf-8") as fw:
            json_data = json.dumps(ori_data, indent=4, ensure_ascii=False)
            fw.write(json_data)
    else:
        raise ValueError("mode:only online, only offline and mix")


if __name__ == "__main__":
    # dataset = "math23k"
    # dataset = "mawps-single-five-fold"
    dataset = "mawps"
    if dataset == "math23k":
        tgt_path = f"./lora_mt/cls_data/classifier_data_{dataset}_online.json"
        tgt_path1 = f"./lora_mt/cls_data/classifier_data_{dataset}_offline.json"
        tgt_final_path = f"./lora_mt/cls_data/classifier_data_{dataset}.json"
        ori_path = f"./data/{dataset}/classifier_train_data.json"
        pred_res_path = f"./lora_mt/cls_data/generated_predictions.json"
        # cs_path = f"./lora_mt/cls_data/keyword_offline_data_{dataset}_train.txt"
        ori_path_valid = f"./data/{dataset}/classifier_valid_data.json"
        dis_path_valid = f"./lora_mt/cls_data/classifier_valid_data_{dataset}_offline.json"
        valid_final_path = f"./lora_mt/cls_data/classifier_valid_data_{dataset}.json"
        sample = True
    elif dataset == "mawps-single-five-fold":
        fold = sys.argv[1]
        tgt_path = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_data_online.json"
        tgt_path1 = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_data_offline.json"
        tgt_final_path = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_data.json"
        ori_path = f"./data/{dataset}/fold{fold}/cls_train.json"
        pred_res_path = f"./lora_mt/cls_data/{dataset}/fold{fold}/generated_predictions.json"
        # cs_path = f"./lora_mt/cls_data/keyword_offline_data_{dataset}_train.txt"
        ori_path_valid = f"./data/{dataset}/fold{fold}/cls_test.json"
        dis_path_valid = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_valid_data_offline.json"
        valid_final_path = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_valid_data.json"
        sample = False
    elif dataset == "mawps":
        fold = sys.argv[1]
        tgt_path = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_data_online.json"
        tgt_path1 = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_data_offline.json"
        tgt_final_path = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_data.json"
        ori_path = f"./data/cv_{dataset}/fold{fold}/train.json"
        pred_res_path = f"./lora_mt/cls_data/{dataset}/fold{fold}/generated_predictions.json"
        # cs_path = f"./lora_mt/cls_data/keyword_offline_data_{dataset}_train.txt"
        ori_path_valid = f"./data/cv_{dataset}/fold{fold}/dev.json"
        dis_path_valid = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_valid_data_offline.json"
        valid_final_path = f"./lora_mt/cls_data/{dataset}/fold{fold}/classifier_valid_data.json"
        sample = False
    else:
        print(f"only dataset name: math23k or mawps-single-five-fold!")
        print("check your dataset name!")
        assert 0 == 1
    online = True
    offline = True
    if online and offline:
        online_update(tgt_path, ori_path, pred_res_path)
        extend_dataset(ori_path_valid, dis_path_valid, valid_final_path,  "mix", sample)
        process_online_data(tgt_path, ori_path)
        extend_dataset(tgt_path, tgt_path1, tgt_final_path, "mix")
    elif offline:
        extend_dataset(ori_path_valid, dis_path_valid, valid_final_path, "mix", sample)
        extend_dataset(ori_path, tgt_path1, tgt_final_path, "mix")
    elif online:
        online_update(tgt_path, ori_path, pred_res_path)
        extend_dataset(ori_path_valid, dis_path_valid, valid_final_path,  "mix", sample)
        process_online_data(tgt_path, ori_path, mode="mix")
        extend_dataset(tgt_path, ori_path, tgt_final_path, "only online")



