import json
import os
import re

from exp_tree import from_postfix_to_infix
from expr_process import from_infix_to_prefix, from_prefix_to_infix, from_infix_to_postfix


def prepare_data_with_answer(data_type="train"):
    data_path = "./"
    data_file = os.path.join(data_path, f"{data_type}23k_processed.json")
    data = json.load(open(data_file))
    with open(f"answer_math23k_processed_{data_type}.txt", "w") as f:
        for d in data:
            idx = d['id']
            text, infix_target = d["original_text"], d["equation"][2:]
            text = text.replace(" ", "")
            # text = map_symbol(text)
            answer = d['answer']

            # try:
            #     infix_target = from_postfix_to_infix(target)  # id 8883 and 17520 are deleted
            # except:
            #     print(idx)
            #     continue

            # infix_target = map_symbol(infix_target)

            # print(infix_target)
            prefix_target = from_infix_to_prefix(infix_target)

            # 断定infix 到 prefix 的转换没有错误

            try:
                new_infix = from_prefix_to_infix(prefix_target)
            except:
                print(idx)
                print(text)
            if idx == "17520" or idx == "8883" or idx == "22085":
                continue

            # new_infix = new_infix.replace("^", "**")

            if data_type == "train" and idx == "9761":  # trian data 中 该条数据答案错误
                answer = 98.33333
            if data_type == "train" and idx == "4303":  # trian data 中 该条数据答案错误
                answer = 1009899
            if data_type == "train" and idx == "12495":  # 同上
                answer = 1111111109
            if data_type == "valid" and idx == "9718":  # 同上
                answer = 5.3817

            # if infix_target == "80千米/小时":
            #     continue
            infix_target = infix_target.replace("[", "(")
            infix_target = infix_target.replace("]", ")")

            f.write("回答以下数学单词问题，并给出解析解：" + text + '\t' + infix_target + "=" + f"{answer}" + '\n')

def prepare_data_with_prefix_answer(data_type="train"):
    data_path = "./"
    data_file = os.path.join(data_path, f"{data_type}23k_processed.json")
    data = json.load(open(data_file))
    with open(f"prefix_answer_math23k_processed_{data_type}.txt", "w") as f:
        for d in data:
            idx = d['id']
            text, infix_target = d["original_text"], d["equation"][2:]
            text = text.replace(" ", "")
            # text = map_symbol(text)
            answer = d['answer']

            # try:
            #     infix_target = from_postfix_to_infix(target)  # id 8883 and 17520 are deleted
            # except:
            #     print(idx)
            #     continue

            # infix_target = map_symbol(infix_target)

            # print(infix_target)
            prefix_target = from_infix_to_prefix(infix_target)

            # 断定infix 到 prefix 的转换没有错误

            try:
                new_infix = from_prefix_to_infix(prefix_target)
            except:
                print(idx)
                print(text)
            if idx == "17520" or idx == "8883" or idx == "22085":
                continue


            new_infix = new_infix.replace("^", "**")

            all_match1 = re.findall(r'\d+\.\d+%', new_infix) # 匹配类似 1.5%
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

            if data_type == "train" and idx == "9761": # trian data 中 该条数据答案错误
                answer = 98.33333
            if data_type == "train" and idx == "4303": # trian data 中 该条数据答案错误
                answer = 1009899
            if data_type == "train" and idx == "12495": # 同上
                answer = 1111111109
            if data_type == "valid" and idx == "9718": # 同上
                answer = 5.3817

            if infix_target == "80千米/小时":
                continue

            try:
                assert abs(eval(new_infix) - answer) < 1e-4
            except:
                print(data_type)

                print(infix_target)
                print(text)
                print(new_infix)
                # print(all_match1)
                # print(all_match2)
                print(eval(new_infix))
                print(answer)
                assert 0 == 1


            f.write("回答以下数学单词问题，并给出先序解："+text + '\t' + prefix_target +"="+ f"{answer}"+'\n')

def prepare_data_with_postfix_answer(data_type="train"):
    data_path = "./"
    data_file = os.path.join(data_path, f"{data_type}23k_processed.json")
    data = json.load(open(data_file))
    with open(f"postfix_answer_math23k_processed_{data_type}.txt", "w") as f:
        for d in data:
            idx = d['id']
            text, infix_target = d["original_text"], d["equation"][2:]
            text = text.replace(" ", "")
            # text = map_symbol(text)
            answer = d['answer']

            # try:
            #     infix_target = from_postfix_to_infix(target)  # id 8883 and 17520 are deleted
            # except:
            #     print(idx)
            #     continue

            # infix_target = map_symbol(infix_target)

            # print(infix_target)
            postfix_target = from_infix_to_postfix(infix_target)

            # 断定infix 到 postfix 的转换没有错误

            try:
                new_infix = from_postfix_to_infix(postfix_target)
            except:
                print(idx)
                print(text)
            if idx == "17520" or idx == "8883" or idx == "22085":
                continue


            new_infix = new_infix.replace("^", "**")

            all_match1 = re.findall(r'\d+\.\d+%', new_infix) # 匹配类似 1.5%
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

            if data_type == "train" and idx == "9761": # trian data 中 该条数据答案错误
                answer = 98.33333
            if data_type == "train" and idx == "4303": # trian data 中 该条数据答案错误
                answer = 1009899
            if data_type == "train" and idx == "12495": # 同上
                answer = 1111111109
            if data_type == "valid" and idx == "9718": # 同上
                answer = 5.3817

            if infix_target == "80千米/小时":
                continue

            try:
                assert abs(eval(new_infix) - answer) < 1e-4
            except:
                print(data_type)

                print(infix_target)
                print(text)
                print(new_infix)
                # print(all_match1)
                # print(all_match2)
                print(eval(new_infix))
                print(answer)
                assert 0 == 1


            f.write("回答以下数学单词问题，并给出后序解："+text + '\t' + postfix_target +"="+ f"{answer}"+'\n')

def math23k_txt2json(file_path, tgt_path, type):
    with open(tgt_path, "w", encoding="utf-8") as fw:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            new_data = []
            for idx, item in enumerate(data):
                d = {}
                q, a = item.strip().split("\t")
                if type == "train":
                    d["question"] = q.replace("回答以下数学单词问题，并给出解析解：", "") + a
                else:
                    d["question"] = q.replace("回答以下数学单词问题，并给出解析解：", "")
                d["answer"] = a
                d["id"] = idx
                d["label"] = 1
                new_data.append(d)
        f.close()
        json_data = json.dumps(new_data, indent=4, ensure_ascii=False)
        fw.write(json_data)




if __name__ == '__main__':
    # prepare_data_with_answer("train")
    # prepare_data_with_answer("valid")
    # prepare_data_with_answer("test")
    # prepare_data_with_prefix_answer("train")
    # prepare_data_with_prefix_answer("valid")
    # prepare_data_with_prefix_answer("test")
    # prepare_data_with_postfix_answer("train")
    # prepare_data_with_postfix_answer("valid")
    # prepare_data_with_postfix_answer("test")
    math23k_txt2json(os.path.join("./answer_math23k_processed_train.txt"),
                     "./classifier_train_data.json", "valid")
    math23k_txt2json(os.path.join("./answer_math23k_processed_valid.txt"),
                     "./classifier_valid_data.json", "valid")
    math23k_txt2json(os.path.join("./answer_math23k_processed_test.txt"),
                     "./classifier_test_data.json", "test")
