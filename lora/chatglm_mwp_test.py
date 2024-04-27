import json
import os
import time

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm

from lora.dataprocess import MWPDatasetForLLM
from lora.metrics import sub_percentage
from lora.misc import get_logits_processor
from model.modeling_chatglm import ChatGLMForConditionalGeneration
from model.tokenization_chatglm import ChatGLMTokenizer
from lora_api import load_model_tokenizer_lora

def api(test_data1, test_data2):

    torch.cuda.manual_seed_all(42)
    model_path = "../model"
    lora_weight_path = "./saved/result_infix_chatglm2_ori2"

    tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
    model = ChatGLMForConditionalGeneration.from_pretrained("../model")

    model = PeftModel.from_pretrained(model, lora_weight_path)
    model.cuda()

    beam = 10
    # gen_kwargs = {
    #             "do_sample": False,
    #             "max_new_tokens": 512 + 1,
    #             "temperature": 1.0, # 0.95
    #             "num_beams": beam,
    #             "num_return_sequences": beam,
    #             "logits_processor": get_logits_processor(),
    #             "return_dict_in_generate": True,
    #             "output_scores": True,
    #             "num_beam_groups": beam,
    #             "diversity_penalty": 4.0,
    #         }
    gen_kwargs = {
                "do_sample": True,
                "max_new_tokens": 512 + 1,
                "temperature": 1.0, # 0.95
                "num_beams": beam,
                "num_return_sequences": beam,
                "logits_processor": get_logits_processor(),
                "return_dict_in_generate": True,
                "output_scores": True,
                # "num_beam_groups": beam,
                # "diversity_penalty": 3.0,
            }

    def inference(test_data):
        with torch.no_grad():
            pred = []
            for input_data in tqdm(test_data):
                # input_ids = tokenizer(input_data[0], return_tensors="pt")
                #
                # print(input_ids)
                # assert 0 == 1
                tmp = list(input_data[0].split("\n"))
                ids = []
                for i in range(len(tmp)-1):
                    q, a = tmp[i].split(">>>")
                    if i > 0:
                        ids += tokenizer.encode(q)[3:]
                    else:
                        ids += tokenizer.encode(q)
                    # ids.append(30910)
                    for token in a:
                        ids.append(tokenizer._convert_token_to_id(token))
                    ids.append(tokenizer._convert_token_to_id("<0x0A>"))
                if len(tmp) == 1:
                    ids += tokenizer.encode(tmp[-1])
                else:
                    ids += tokenizer.encode(tmp[-1])[3:]
                ids = tokenizer.get_prefix_tokens() + ids[2:]
                # print(tokenizer.convert_ids_to_tokens(ids))
                # print(tokenizer.decode(ids))
                # print(tokenizer.convert_ids_to_tokens(tokenizer.encode(input_data[0], add_special_tokens=False)))
                # assert 0 == 1
                attention_mask = [1 for _ in range(len(ids))]

                ids_tensor = torch.LongTensor(ids)
                attention_mask = torch.LongTensor(attention_mask)
                input_ids = {"input_ids":ids_tensor.cuda().unsqueeze(dim=0), "attention_mask":attention_mask.cuda().unsqueeze(dim=0)}
                # input_ids.to("cuda")
                output = model.generate(**input_ids, **gen_kwargs)
                score = np.exp(output["sequences_scores"].cpu().numpy()[0])
                pred_token = tokenizer.decode(output["sequences"][0].cpu()[len(input_ids["input_ids"][0]):], skip_special_tokens=False)
                pred.append([pred_token, score])
                # print(pred)
        torch.cuda.empty_cache()
        return pred
    return inference(test_data1), inference(test_data2)

    # print(output)


    # rst = tokenizer.batch_decode(output)
    # for item, score in zip(output["sequences"], output["sequences_scores"]):
    #     print(" ".join(tokenizer.convert_ids_to_tokens(item[len(input_ids["input_ids"][0]):], skip_special_tokens=False)))
    #     print(tokenizer.decode(item[len(input_ids["input_ids"][0]):], skip_special_tokens=False))
    #     print(torch.exp(score).item())
    #     print("\n")

def compute_metric(test_data, pred, data_type):
    score_dict = 0
    for label, pred in zip(test_data, pred):
        label_eq, labels_ans = label[1].split('=')
        try:
            if data_type == "sub":
                _, pred = pred.split(">>>")
                _, pred_eq, pred_ans = pred.split('=')
            elif data_type == "ori":
                pred_eq, pred_ans = pred.split('=')

            else:
                raise TypeError("data type must be 'ori' or 'sub'!")


            # 替换"^"为"**", 以及替换"%"
            pred_eq = pred_eq.replace("^", "**")
            pred_eq = sub_percentage(pred_eq)

            ans = eval(pred_eq)
            if abs(ans - eval(labels_ans)) < 1e-4:
                score_dict += 1
                print("Right Expression:" + pred)
                print("-----------------")
            else:
                score_dict += 0
                print("Wrong Expression:" + pred)
                print("-----------------")

        except:
            # eval() function not sppoort "%"计算，if "%" leads to excpetion, we just consider pred_ans.
            if abs(eval(pred_ans) - eval(labels_ans)) < 1e-4:
                score_dict += 1
                print("Right Answer but Expression can't be calculated:" + pred)
                print("-----------------")
            else:
                score_dict += 0
                print("Wrong Answer and Expression can't be calculated:" + pred)
                print("-----------------")
            # score_dict["accuracy"].append(0)
        finally:
            continue
    print(score_dict)
    return score_dict/len(test_data)


def score_contrast(sft_pred, fs_pred):
    assert len(sft_pred) == len(fs_pred)
    final_pred = []
    for sft, fs in zip(sft_pred, fs_pred):
        # calculate average probability over length of the decoding string
        # if sft[1]/(len(sft[0])+1) >= fs[1]/(len(fs[0])+1):
        if sft[1] >= fs[1]:
            final_pred.append(sft[0])
        else:
            final_pred.append(fs[0])
    return final_pred


def readfile(file_path):
    test_data = []
    if file_path.endswith("json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                test_data.append([item["prompt"], item["answer"]])
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                prompt, label = line.strip().split("\t")
                test_data.append([prompt, label])

    return test_data


def final_test(sft_test_file, icl_test_file):
    if not os.path.exists("./icl"):
        os.makedirs("./icl")
    # assert 0 == 1
    test_data = readfile(sft_test_file)
    sft_pred, icl_pred = api(test_data, readfile(icl_test_file))
    sft_acc = compute_metric(test_data, [_[0] for _ in sft_pred], "ori")
    print(f"sft_acc:{sft_acc}")
    icl_acc = compute_metric(test_data, [_[0] for _ in icl_pred], "ori")
    print(f"icl_acc:{icl_acc}")
    final_pred = score_contrast(sft_pred, icl_pred)
    acc = compute_metric(test_data, final_pred, "ori")
    print(f"final_acc:{acc}")

    with open("./icl/result.json", "w", encoding="utf-8") as fw:
        data = []
        for p1, p2, ori in zip(sft_pred, icl_pred, test_data):
            data.append({"sft pred": p1[0], "sft score": f"{p1[1]/(len(p1[0])+1)}", "sft R_W": compute_metric([ori], [p1[0]], "ori")
                         , "icl pred": p2[0], "icl score": f"{p2[1]/(len(p2[0])+1)}", "icl R_W": compute_metric([ori], [p2[0]], "ori")
                         , "Q": ori[0], "A": ori[1]})
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        fw.write(json_data)


if __name__ == "__main__":
    sft_path = "../data/math23k/answer_math23k_processed_test.txt"
    icl_path = "../data/math23k/icl_test.json"
    final_test(sft_path, icl_path)




