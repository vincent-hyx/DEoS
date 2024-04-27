import json
import re

from data.math23k.subproblem import subprobelm_change


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        qa_data = []
        data = f.readlines()
        for item in data:
            q, a = item.strip("\n").split("\t")
            qa_data.append([q, a])
    return qa_data


def extract(qa_data, tgt_path):
    keyword = ["几", "多少", "="]
    delim_sym = ["，", ".", "。", "．"]
    end_sym = [".", "。", "?", "？", "．"]
    percent_filed = ["百分之几", "几分之几"]
    number_sym = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    new_data = []
    for item in qa_data:
        q, a = item[0], item[1]
        extract_q = ""
        for k in keyword:
            tgt = re.search(k, q)
            if tgt is not None:
                l = tgt.span()[0]
                if q[l-1] in number_sym:
                    break
                while q[l] not in delim_sym and l > 0:
                    l -= 1
                l += 1
                extract_q = q[l:]
                if k == "=":
                    extract_q = extract_q.replace(k, "=x")
                elif k == keyword[0]:
                    extract_q = extract_q.replace(percent_filed[0], "x")
                    extract_q = extract_q.replace(percent_filed[1], "x")
                    extract_q = extract_q.replace(k, "x")
                else:
                    extract_q = extract_q.replace(k, "x")
                if extract_q[-1] in end_sym:
                    extract_q = extract_q[:-1]
                break
            else:
                continue

        new_data.append({"q": q, "a": a, "extract_q": extract_q})

    with open(tgt_path, 'w', encoding="utf-8") as fw:
        for data in new_data:
            expr_ans = data["a"]
            text = data["q"]
            ex_q = data["extract_q"]
            process_response = subprobelm_change(expr_ans)
            if ex_q != "":
                if process_response is not None:
                    response = f"假设{ex_q},那么{process_response}>>>x={expr_ans}\n"
                    fw.write(text + "\t" + response)
                else:
                    response = f"假设{ex_q},那么>>>x={expr_ans}\n"
                    fw.write(text + "\t" + response)
            else:
                print(text)
                print("输入！")
                human_response = input().strip()
                if human_response != "1":
                    response = f"{human_response},那么>>>答案={expr_ans}\n"
                    fw.write(text + "\t" + response)
                else:
                    response = f"根据题意,那么>>>答案={expr_ans}\n"
                    fw.write(text + "\t" + response)


    # with open(tgt_path, "w", encoding="utf-8") as fw:
    #     data = json.dumps(new_data, indent=4, ensure_ascii=False)
    #     fw.write(data)
def post_process(ex_path, src_path, tgt_path):
    ex_q_list = []
    with open(ex_path, "r", encoding="utf-8") as fr:
        data = fr.readlines()
        for line in data:
            text, response = line.strip().split("\t")
            ex_q, _ = response.split(",那么")
            ex_q_list.append(ex_q)

    with (open(tgt_path, "w", encoding="utf-8") as fw):
        with open(src_path, "r", encoding="utf-8") as frr:
            src_data = frr.readlines()
            for line, ex_q in zip(src_data, ex_q_list):
                text, expr_ans = line.strip().split("\t")
                process_response = subprobelm_change(expr_ans)
                if ex_q != "根据题意":
                    if ex_q[:2] != "假设":
                        ex_q = "假设" + ex_q
                if process_response is not None:
                    response = f"{ex_q},那么{process_response}>>>x={expr_ans}\n"
                    fw.write(text + "\t" + response)
                else:
                    response = f"{ex_q},那么>>>x={expr_ans}\n"
                    fw.write(text + "\t" + response)



if __name__ == '__main__':
    tgt_path = "./extract_q_trian.txt"
    # extract(load_data("answer_math23k_processed_train.txt"), tgt_path)
    post_process(tgt_path, "answer_math23k_processed_train.txt", "complete_extract_train_final.txt")
    tgt_path = "./extract_q_valid.txt"
    # extract(load_data("answer_math23k_processed_valid.txt"), tgt_path)
    post_process(tgt_path, "answer_math23k_processed_valid.txt", "complete_extract_valid_final.txt")
    tgt_path = "./extract_q_test.txt"
    # extract(load_data("answer_math23k_processed_test.txt"), tgt_path)
    post_process(tgt_path, "answer_math23k_processed_test.txt", "complete_extract_test_final.txt")
