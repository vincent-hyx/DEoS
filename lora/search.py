import torch
from peft import PeftModel

from lora.misc import get_logits_processor
from model.modeling_chatglm import ChatGLMForConditionalGeneration
from model.tokenization_chatglm import ChatGLMTokenizer

torch.cuda.manual_seed_all(42)
model_path = "../model"
# lora_weight_path = "./saved/result_infix_chatglm2_token"
# lora_weight_path = "../lora_classifier/saved/result_classifier"
lora_weight_path = "./saved/result_infix_chatglm2_ori2"
tokenizer = ChatGLMTokenizer.from_pretrained(model_path)
model = ChatGLMForConditionalGeneration.from_pretrained("../model")

model = PeftModel.from_pretrained(model, lora_weight_path)
model.cuda()
# prompt = "回答以下数学单词问题，并给出解析解："
# example1 = "一只绵羊的质量是58千克，一头牛的质量是这只绵羊质量的9倍多20千克，这头牛的质量=多少千克？\n58*9+20=542.0\n"
# example2 = "学校有教师45人，学生的人数比教师人数的14倍多28人，学校有学生和教师一共多少人？\n45*14+28+45=703.0\n"
# example3 = "养鸡专业户王大伯去年养鸡1090只，今年养鸡只数增加到去年的3倍．今年养鸡多少只？\n1090*3=3270.0\n"
# example4 = "少先队员种柳树40棵，种的杨树的棵数比柳树棵数的3倍多15棵．少先队员种杨树和柳树共多少棵？\n40+(40*3+15)=175.0\n"
# example5 = "在新农村建设中大西村要修一条水泥路，已修了180米，剩下的比已修的3倍多16米，这条路全长多少米？\n180*3+16+180=736.0\n"
# example6 = "粮店上午运进120大米袋，下午比上午的2倍多40袋，这天共运进大米多少袋．\n120*2+40+120=400.0\n"
# all_example = example2 + example3 + example4 + example5 + example6
# query = "回答以下数学单词问题，并给出解析解：甲乙两个修路队合修完一条公路，甲队修的公路长增加到2倍多30米，才和乙对修的一样多，甲队修了250米．乙队修了多少米？"
query = "回答以下数学单词问题，并给出解析解：甲乙两个修路队合修完一条公路，甲队修的公路长增加到2倍多30米，才和乙修的一样多，甲队修了250米．这条公路长多少米？"
# query = "甲乙两个修路队合修完一条公路，甲队修了250米，乙队修的是甲队的2倍多30米．这条公路长多少米？"
# query = "回答以下数学单词问题，并给出解析解：甲乙两个修路队合修完一条公路，甲队修的公路长增加到2倍多30米，才和乙队修的一样多，甲队修了250米．乙队修了多少米？"
# query = "一根钢材长18.4米，锯下3.8米后，剩下的比锯下的长几米？"

# example1 = "回答以下数学单词问题，并给出解析解：李叔叔在一块正方形菜地的4周围篱笆，每隔3米打一根木桩，一共打了20根木桩，这个正方形的边长=多少米？\n20*3=60.0\n"
# example2 = "回答以下数学单词问题，并给出解析解：一块正方形地毯的4周需要镶一条花边．地毯的边长为4米，求所需花边长度？\n4*4=16.0\n"
# example3 = "回答以下数学单词问题，并给出解析解：沿一个长方形操场4周种树，每隔4米种一棵，一共种了30棵．这个操场的周长=多少米．\n30*4=120.0\n"
# example4 = "回答以下数学单词问题，并给出解析解：在圆形操场的4周每隔3米插上彩旗，共插上80面，这个圆形操场的周长=多少米．\n80*3=240.0\n"
# example5 = "回答以下数学单词问题，并给出解析解：一个圆形养鱼池的周长是500米，在4周每隔5米栽一棵柳树，共需栽柳树多少棵．\n500/5=100.0\n"
# all_example = example1 + example2 + example3 + example4 + example5
# query = "回答以下数学单词问题，并给出解析解：两根绳子，甲根长9米，乙根长6米，各剪去同样长的一段后，乙根的长是甲根的(3/5)，甲根剪去多少米．"
# input = all_example + query
input = query
# input = "回答以下数学单词问题，并给出解析解：甲乙两个修路队合修完一条公路，乙队修的公路比甲队修的的2倍多30米，甲队修了250米．甲乙一共修了多少米？"
# input = "回答以下数学单词问题，并给出解析解：学校有教师45人，学生的人数比教师人数的14倍多28人，学校有学生和教师一共多少人？"

# input = "回答以下数学单词问题，并给出解析解：商店为了搬运方便，把4瓶啤酒捆在一起，啤酒瓶的外直径为10厘米，用绳子捆一周，至少要用多长的绳子？\n10*4+3.14*10=71.4\n回答以下数学单词问题，并给出解析解：篮球场的长是28米，宽是15米，它的面积=多少平方米．\n28*15=420.0\n回答以下数学单词问题，并给出解析解：一个圆形广场的周围栽了48棵树，相邻两棵树之间的距离是5米，这个广场的周长=多少米？\n5*48=240.0\n回答以下数学单词问题，并给出解析解：一块正方形地毯的4周需要镶一条花边．地毯的边长为4米，求所需花边长度？\n4*4=16.0\n回答以下数学单词问题，并给出解析解：一张长方形硬纸板长12厘米，宽9厘米．在这个长方形中，剪去一个最大的正方形，剩下的小长方形的面积=？\n9*(12-9)=27.0\n回答以下数学单词问题，并给出解析解：把长为18厘米、宽是12厘米的正方形纸片剪成边长是3厘米的正方形纸片，共可剪成多少个这样的小正方形？\n(18/3)*(12/3)=24.0\n回答以下数学单词问题，并给出解析解：在新农村建设中大西村要修一条水泥路，已修了180米，剩下的比已修的3倍多16米，这条路全长多少米？\n180*3+16+180=736.0\n回答以下数学单词问题，并给出解析解：汽车的速度是35千米/时，一列火车的速度比汽车多30千米/时，飞机的速度是火车的4倍，这架飞机的速度=？\n(35+30)*4=260.0\n回答以下数学单词问题，并给出解析解：在一正方形花池的4周栽了44棵柳树，每两棵柳树之间的间隔是20米，这个正方形的周长=多少米？"
input_ids = tokenizer(input, return_tensors="pt")
print(tokenizer.decode(input_ids["input_ids"][0]))
input_ids.to("cuda")
print(input_ids)
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

# print(tokenizer.special_tokens)
# assert 0 == 1
def constrain_decode_vocab(batch_idx, prefix_beam):
    constrain_set_op = ["+", "-", "*", "/", "^", "="]
    keywords = tokenizer._convert_token_to_id("#")
    constrain_set_num = [f"{_}" for _ in range(10)]
    constrain_set_other = ["(", ")", "."]
    constrain_set_special_token = ["<bos>", "<eos>", "sop", "eop", "</s>"]

    all_vocab = constrain_set_num + constrain_set_op + constrain_set_other + constrain_set_special_token
    if prefix_beam[-1] == keywords:
        if prefix_beam[-2] in tokenizer.convert_tokens_to_ids(constrain_set_op) and prefix_beam[-3] == keywords:
            return tokenizer.convert_tokens_to_ids(constrain_set_num)
        else:
            restricted_vocab = tokenizer.convert_tokens_to_ids(constrain_set_op)
            return restricted_vocab
    else:
        return tokenizer.convert_tokens_to_ids(all_vocab)


gen_kwargs = {
            "do_sample": True,
            "max_new_tokens": 512 + 1,
            "temperature": 1.0, # 0.95
            "num_beams": beam,
            "num_return_sequences": beam,
            "return_dict_in_generate": True,
            "output_scores": True,
            "logits_processor": get_logits_processor(),
            # "prefix_allowed_tokens_fn": constrain_decode_vocab,
        }
# gen_kwargs = {
#             "do_sample": True,
#             "max_new_tokens": 512 + 1,
#             "temperature": 1.0, # 0.95
#             "top_p": 1.0,
#         }

output = model.generate(**input_ids, **gen_kwargs)
print(output)


# print(tokenizer.batch_decode(output))
for item, score in zip(output["sequences"], output["sequences_scores"]):
    print(" ".join(tokenizer.convert_ids_to_tokens(item[len(input_ids["input_ids"][0]):], skip_special_tokens=False)))
    print(tokenizer.decode(item[len(input_ids["input_ids"][0]):], skip_special_tokens=False))
    print(torch.exp(score).item())
    print("\n")


