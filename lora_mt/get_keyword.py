import json
import re
import time

import zhipuai
from tqdm import tqdm
from update_train_cls import readfile
from sklearn.model_selection import train_test_split

zhipuai.api_key ="a0d6ee910056d80650f57743d60b0419.CtaN4pKho1V0mxRe"#填写控制台中获取的 APIKey 信息

from zhipuai import ZhipuAI

def getText(role, content, text=[]):
    # role 是指定角色，content 是 prompt 内容
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text


def access_glm(content):
    # content = "请提取{" + data + "}中的关键字，并把结果放在{}中"
    client = ZhipuAI(api_key="a0d6ee910056d80650f57743d60b0419.CtaN4pKho1V0mxRe")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": content},
        ],
    )
    ans = response.choices[0].message.content
    return ans


def access_tfidf():
    from jieba import analyse
    # 引入TF-IDF关键词抽取接口
    tfidf = analyse.extract_tags

    # 原始文本
    text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
            是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
            线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
            线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
            同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"

    # 基于TF-IDF算法进行关键词抽取
    keywords = tfidf(text)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword)


def access_textrank():
    from jieba import analyse
    # 引入TextRank关键词抽取接口
    textrank = analyse.textrank

    # 原始文本
    text = "一个收购站运走鸡蛋60箱和鸭蛋48箱，每箱鸡蛋重40千克，每箱鸭子重35千克．一共运走禽蛋多少千克？"

    # 基于TextRank算法进行关键词抽取
    keywords = textrank(text)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword)


def keyword2file(ori_path, tgt_path):

    with open(ori_path, "r", encoding="utf-8") as fr:
        data = []
        lines = fr.readlines()
        for li in lines:
            q, a = li.strip().split("\t")
            q = q.replace("回答以下数学单词问题，并给出解析解：", "")
            data.append([q, a])
    with open(tgt_path, "w", encoding="utf-8") as fw:
        for item in tqdm(data):
            content = "请提取{" + item[0] + "}中的关键字，并把结果放在{}中"
            keyword = access_glm(content)
            # time.sleep(0.1)
            q = item[0]
            a = item[1]
            fw.write(f"{q}\t{a}\t{keyword}\n")

# def split_sample(tgt_path):
#     data = []
#     with open(tgt_path, "r", encoding="utf-8") as fr:
#         lines = fr.readlines()
#         for li in lines:
#             try:
#                 q, a, keyword = li.strip().split("\t")
#             except:
#                 continue
#             data.append([q, a, keyword])
#     new_data = process_keyword(data)
#     print(len(new_data))
#     train_data, valid_data = train_test_split(new_data, shuffle=True, train_size=0.5, random_state=42)
#     print(len(train_data))
#     train_data_cls = []
#     for item in tqdm(train_data):
#         q, a, key = item[0], item[1], item[2]
#         for k in key:
#             cls_data = {}
#             q1 = q.replace(k, "[MASK]")
#
#             print(f"原句\n{q}\n改变后:\n{q1}\n语义是否反转？(0 或 1)   {k}")
#             res = input()
#             if eval(res) == 0:
#                 cls_data["label"] = 0
#             elif eval(res) == 1:
#                 cls_data["label"] = 1
#             else:
#                 continue
#             cls_data["question"] = q1
#             cls_data["id"] = "offline"
#             cls_data["answer"] = a
#             train_data_cls.append(cls_data)
#     new_tgt = tgt_path.replace(".txt", "_label.json")
#     with open(new_tgt, "w", encoding="utf-8") as fw:
#         json_data = json.dumps(train_data_cls, indent=4, ensure_ascii=False)
#         fw.write(json_data)
#     fw.close()
#     new_tgt1 = new_tgt.replace(".json", "_unlabel.json")
#     valid_data_cls = []
#     for item in valid_data:
#         cls_data = {}
#         q, a, key = item[0], item[1], item[2]
#         cls_data["id"] = "offline"
#         cls_data["answer"] = a
#         cls_data["label"] = -2
#         for k in key:
#             q1 = q.replace(k, "[MASK]")
#             cls_data["question"] = q1
#             valid_data_cls.append(cls_data)
#     with open(new_tgt1, "w", encoding="utf-8") as fw:
#         json_data = json.dumps(valid_data_cls, indent=4, ensure_ascii=False)
#         fw.write(json_data)

def generate_sample(tgt_path, dataset_name="math23k"):
    data = []
    with open(tgt_path, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for li in lines:
            try:
                q, a, keyword = li.strip().split("\t")
            except:
                continue
            data.append([q, a, keyword])
    new_data = process_keyword(data)
    # print(len(new_data))
    # train_data, valid_data = train_test_split(new_data, shuffle=True, train_size=0.5, random_state=42)
    # print(len(train_data))
    tgt_path1 = tgt_path.replace(".txt", f"_offline_data_{dataset_name}.txt")
    with open(tgt_path1, "w", encoding="utf-8") as fw:

        for item in tqdm(new_data):
            q, a, key = item[0], item[1], item[2]
            for k in key:
                q1 = q.replace(k, "[MASK]")
                content = f"判断以下两句话的语义是否相同，只回答相同或不相同，不要解释:{q}和{q1}"
                ans = access_glm(content)
                # print(ans)

                if ans == "相同":
                    label = 1
                elif ans == "不相同":
                    label = 0
                else:
                    continue
                fw.write(f"{q1}\t{a}\t{label}\n")
                time.sleep(0.5)







def process_keyword(data:list):
    new_data = []
    for item in data:
        keyword = item[-1]
        keyword = keyword.replace("{", "")
        keyword = keyword.replace("}", "")
        keyword = keyword.replace(",", " ")
        keyword = keyword.replace("，", " ")
        tmp_list = list(keyword.split(" "))
        keyword_list = [_.replace(" ", "") for _ in tmp_list if _ != " " and _ != ""]
        new_data.append([item[0], item[1], keyword_list])
    return new_data

def split_sample(ori_valid_path, train_path, valid_path):
    data = []
    with open(train_path, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for li in lines:
            q, a, label = li.strip().split("\t")
            data.append([q, a, label])
    train_data, valid_data = train_test_split(data, shuffle=True, train_size=0.9, random_state=42)
    new_train_path = train_path.replace(".txt", "_train.txt")
    with open(new_train_path, "w", encoding="utf-8") as fw:
        for item in tqdm(train_data, desc="cs train samples"):
            q, a, label = item[0], item[1], item[2]
            fw.write(f"{q}\t{a}\t{label}\n")
    # ori_valid = readfile(ori_valid_path)
    cs_valid = []
    for item in tqdm(valid_data, desc="cs valid sample"):
        cls_data = {}
        cls_data["question"] = item[0]
        cls_data["answer"] = item[1]
        cls_data["id"] = "offline"
        cls_data["label"] = eval(item[2])
        cs_valid.append(cls_data)
    with open(valid_path, "w", encoding="utf-8") as fw:
        json_data = json.dumps(cs_valid, indent=4, ensure_ascii=False)
        fw.write(json_data)





if __name__ == "__main__":
    # ori_path = "./data/math23k/answer_math23k_processed_train.txt"
    tgt_path = "./lora_mt/cls_data/keyword.txt"
    # keyword2file(ori_path, tgt_path)
    # textRank()
    # generate_sample(tgt_path)
    ori_valid_path = "./data/math23k/classifier_valid_data.json"
    train_path = "./lora_mt/cls_data/keyword_offline_data_math23k.txt"
    valid_path = "./lora_mt/cls_data/classifier_valid_data_math23k.json"
    split_sample(ori_valid_path, train_path, valid_path)



