import os
ORDER_DICT = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}


def from_infix_to_postfix(expression):
    if isinstance(expression, str):
        expression = expr2list(expression)
    st = list()
    res = list()
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in ORDER_DICT:
            while len(st) > 0 and st[-1] not in ["(", "["] and ORDER_DICT[e] <= ORDER_DICT[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return " ".join(res)


def from_infix_to_prefix(expression):
    if isinstance(expression, str):
        expression = expr2list(expression)
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return ' '.join(res)



def expr2list(expr):
    sym_set = ['+', '-', '*', '/', '^', '(', ')', '[', ']']
    start = 0
    res = []
    while start < len(expr):
        if expr[start] not in sym_set:
            end = start + 1
            while end < len(expr) and expr[end] not in sym_set:
                end += 1
            # if end < len(expr):
            #     res.append(expr[start:end])
            # else:
            #     res.append(expr[start:end+1])
            res.append(expr[start:end])
            start = end
        else:
            res.append(expr[start])
            start += 1
    return res

def from_prefix_to_infix(expr:str):
    expression = list(expr.split(" "))
    sym_set = ["+", '-', '*', '/', '^']
    stack = []
    for e in expression:
        stack.append(e)
        if e not in sym_set:
            while len(stack) > 1 and stack[-2] not in sym_set:
                op_b = stack.pop()
                op_a = stack.pop()
                op = stack.pop()
                if len(stack) != 0:
                    res = f"({op_a}{op}{op_b})"
                else:
                    res = f"{op_a}{op}{op_b}"
                stack.append(res)
    while len(stack) > 1:
        op_b = stack.pop()
        op_a = stack.pop()
        op = stack.pop()
        if len(stack) != 0:
            res = f"({op_a}{op}{op_b})"
            print(res)
        else:
            res = f"{op_a}{op}{op_b}"
        stack.append(res)
    assert len(stack) == 1
    return stack[0]









# expr = "30*(1-(1.5/5))+5"
#
#
# expr_list = expr2list(expr)
#
#
# prefix = from_infix_to_prefix(expr_list)
#
# print(prefix)
#
# print(eval(from_prefix_to_infix(prefix)))
# str = "1+2%"
# str_1 = "1+2"
# m = re.match(r'\d+%', str_1)



