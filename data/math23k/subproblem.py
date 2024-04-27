import re


def sub_percentage(new_infix): # 先把 '小数%' 替换，再替换"整数%"， 且每次替换只替换一次
    all_match1 = re.findall(r'\d+\.\d+%', new_infix)  # 匹配类似 1.5%
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
    return new_infix
def subprobelm_change(expr_ans):
    expr, ans = expr_ans.split("=")
    match = re.findall(r"\(\([\d\+\-*/]+\).\([\d\+\-*/]+\)\)", expr)
    if len(match) == 1 and len(match[0]) == len(expr):
        print(match)
        expr = expr.replace(match[0], match[0][1:-1])
    # 去掉最外层多余括号
    flag = False
    if expr[0] == "(" and expr[-1] == ")":
        old_expr = sub_percentage(expr).replace("^", "**")
        new_expr = expr[1:-1]
        try:
            assert eval(sub_percentage(new_expr).replace("^", "**")) == eval(old_expr)
        except:
            flag = True
        finally:
            if flag is False:
                expr = new_expr


    op = [] # 从左向右记录每个op的位置及所在子问题的级别
    for index, char in enumerate(expr):
        if char in "+-*/":
            op_dict = {}
            op_dict["index"] = index
            sub_level = 0
            if char in "*/":
                sub_level += 0.5
            for i in range(index, -1, -1):
                if expr[i] == "(":
                    sub_level += 1
                if expr[i] == ")":
                    sub_level -= 1
            op_dict["level"] = sub_level
            op.append(op_dict)
    if len(op) >= 2 :
        op_index = -1
        cur_level = 1000
        for item in op:
            if item["level"] < cur_level:
                op_index = item["index"]
                cur_level = item["level"]
            elif item["level"] == cur_level:
                if item["index"] > op_index:
                    op_index = item["index"]
            else:
                pass

        if expr[op_index] == "+":
            left_expr = sub_percentage(f"{ans}-{expr[op_index + 1:]}").replace("^", "**")
            right_expr = sub_percentage(expr[:op_index]).replace("^", "**")
            assert abs(eval(left_expr) - eval(right_expr)) < 1e-4
            return 'x-' + expr[op_index + 1:] + '=' + expr[:op_index]
        if expr[op_index] == "-":
            left_expr = sub_percentage(f"{ans}+{expr[op_index + 1:]}").replace("^", "**")
            right_expr = sub_percentage(expr[:op_index]).replace("^", "**")
            print(left_expr)
            print(right_expr)
            print(expr_ans)
            assert abs(eval(left_expr) - eval(right_expr)) < 1e-4
            return 'x+' + expr[op_index + 1:] + '=' + expr[:op_index]
        if expr[op_index] == "*":
            left_expr = sub_percentage(f"{ans}/{expr[op_index + 1:]}").replace("^", "**")
            right_expr = sub_percentage(expr[:op_index]).replace("^", "**")
            print(left_expr, right_expr, expr)
            print(expr[op_index + 1:])
            assert abs(eval(left_expr) - eval(right_expr)) < 1e-4
            return 'x/' + expr[op_index + 1:] + '=' + expr[:op_index]
        if expr[op_index] == "/":
            left_expr = sub_percentage(f"{ans}*{expr[op_index + 1:]}").replace("^", "**")
            right_expr = sub_percentage(expr[:op_index]).replace("^", "**")
            print(left_expr)
            print(right_expr)
            print(expr_ans)
            print(eval(left_expr))
            print(eval(right_expr))
            print(abs(eval(left_expr) - eval(right_expr)))
            if ans == "5.3817":
                return 'x*' + expr[op_index + 1:] + '=' + expr[:op_index]
            assert abs(eval(left_expr) - eval(right_expr)) < 1e-4
            return 'x*' + expr[op_index + 1:] + '=' + expr[:op_index]
        else:
            return None


def further_decompose(pr:str, new_var:str):
    sub_str, _ = pr.split("=")
    time = 0
    for char in sub_str:
        if char in "+-*/":
            time += 1
    if time > 1:
        sub_str = sub_str[2:]
        sub_str_value = eval(sub_percentage(sub_str).replace("^", "**"))  # uses to verif
        sub_str = sub_str + "=" + f"{sub_str_value}"
        new_sub = subprobelm_change(sub_str)
        if new_sub is not None:
            new = new_sub.replace("x", new_var, 1)
            return new, sub_str.split("=")[0]
        else:
            return None, None
    else:
        return None, None

def subproblem_process(type:str):
    file_path = f"./answer_math23k_processed_{type}.txt"
    tgt_file_path = f"./answer_math23k_processed_{type}_decompose_sub.txt"
    with open(tgt_file_path, 'w', encoding="utf-8") as fw:
        with open(file_path, "r", encoding="utf-8") as f:
            all_data = f.readlines()
            for data in all_data:
                text, expr_ans = data.strip().split("\t")
                process_response = subprobelm_change(expr_ans)
                if process_response is not None:
                    print(process_response)
                    second_step, second = further_decompose(process_response, "y")
                    if second is not None:
                        response = f"假设最终答案为x，中间步骤为y,求解y={second},那么>>>x={expr_ans}\n"
                        fw.write(text + "\t" + response)
                    else:
                        response = f"直接求解最终答案x,那么>>>x={expr_ans}\n"
                        fw.write(text + "\t" + response)
                else:
                    response = f"直接求解最终答案x,那么>>>x={expr_ans}\n"
                    fw.write(text + "\t" + response)
# 等价问题
# def subproblem_process(type:str):
#     file_path = f"./answer_math23k_processed_{type}.txt"
#     tgt_file_path = f"./answer_math23k_processed_{type}_decompose_twostep.txt"
#     with open(tgt_file_path, 'w', encoding="utf-8") as fw:
#         with open(file_path, "r", encoding="utf-8") as f:
#             all_data = f.readlines()
#             for data in all_data:
#                 text, expr_ans = data.strip().split("\t")
#                 process_response = subprobelm_change(expr_ans)
#                 if process_response is not None:
#                     print(process_response)
#                     second_step, second = further_decompose(process_response, "y")
#                     if second_step is not None:
#                         first_step = process_response.replace(second, "y", 1)
#                         response = f"假设最终答案为x，中间变量为y,那么由{second_step}得到y={second},接下来{first_step}>>>x={expr_ans}\n"
#                         fw.write(text + "\t" + response)
#                     else:
#                         response = f"假设最终答案为x,那么{process_response}>>>x={expr_ans}\n"
#                         fw.write(text + "\t" + response)
#                 else:
#                     response = f"直接求解最终答案x,那么>>>x={expr_ans}\n"
#                     fw.write(text + "\t" + response)

# single step decompose
# def subproblem_process(type:str):
#     file_path = f"./answer_math23k_processed_{type}.txt"
#     tgt_file_path = f"./answer_math23k_processed_{type}_decompose.txt"
#     with open(tgt_file_path, 'w', encoding="utf-8") as fw:
#         with open(file_path, "r", encoding="utf-8") as f:
#             all_data = f.readlines()
#             for data in all_data:
#                 text, expr_ans = data.strip().split("\t")
#                 process_response = subprobelm_change(expr_ans)
#                 if process_response is not None:
#                     response = f"假设最终答案为x,那么{process_response}>>>x={expr_ans}\n"
#                     fw.write(text + "\t" + response)
#                 else:
#                     response = f"假设最终答案为x,那么>>>x={expr_ans}\n"
#                     fw.write(text + "\t" + response)
# multitask
# def subproblem_process(type:str):
#     file_path = f"./answer_math23k_processed_{type}.txt"
#     tgt_file_path = f"./answer_math23k_processed_{type}_decompose1.txt"
#     with open(tgt_file_path, 'w', encoding="utf-8") as fw:
#         with open(file_path, "r", encoding="utf-8") as f:
#             all_data = f.readlines()
#             for data in all_data:
#                 text, expr_ans = data.strip().split("\t")
#                 process_response = subprobelm_change(expr_ans)
#                 if process_response is not None:
#                     res_prompt, res_answer = process_response.split("=")
#                     fw.write(text + "假设最终答案为x,那么" + res_prompt + "=" + "\t" + res_answer + "\n")
#                 # else:
#                 #     response = f"假设最终答案为x,那么>>>x={expr_ans}\n"
#                 #     fw.write(text + "\t" + response)


if __name__ == '__main__':
    subproblem_process("train")
    subproblem_process("valid")
    subproblem_process("test")











