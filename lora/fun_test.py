# print(eval("1-15%"))
# from sympy import *
# z = Rational(1, 2) # 构造分数 1/2
# print(z)
# import os
# path1 = r"../lora/saved/result1026"
# f = os.listdir(path1)
# all_checkpoint = []
# for dirnames in f:
#     if "checkpoint" in dirnames:
#         all_checkpoint.append(dirnames)
# all_checkpoint.sort(key=lambda x:int(x[10:]), reverse=True)
# print(all_checkpoint)
# import os
#
# with open(os.path.join('../lora', "result_acc.txt"), 'w') as f:
#     f.write("2\n")
#     f.write(f"\nbest_epoch:1||best_acc:1")
# import torch
# print(torch.cuda.is_available())
# import re
#
# new_infix = "5% - 5%"
# # all_match1 = re.findall(r'\d+\.\d+%', new_infix)  # 匹配类似 1.5%
# all_match2 = re.findall(r'\d+%', new_infix)  # 匹配类似 1.5%
# print(all_match2)
# import torch
# import torch.nn.functional as F
# F.ctc_loss()
# torch.ctc_loss()
# import re
# s = "we're"
# print(s)
# new_s = re.findall(r".'", s)
# for item in new_s:
#     rel_s = item[0] + " " + item[1]
#     s = s.replace(item, rel_s)
# print(s)
# s = "sss"
# print(s[1:-1])
# print(eval("((80/100)+((3)/(3+2))-1)"))
import torch

# 创建一个3维张量，尺寸为(2, 3, 4)
tensor_3d = torch.tensor([[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]],

                          [[13, 14, 15, 16],
                           [17, 18, 19, 20],
                           [21, 22, 23, 24]]])

# 创建一个2维索引张量，尺寸为(2, 2)
index_tensor = torch.tensor([[0],[1]])
print(tensor_3d[[[0,1]], [[0,1]], :])

