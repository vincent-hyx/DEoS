
def read_txt(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def ana_wrong(ori, deos):
    assert len(ori) == len(deos)
    cnt = 0
    for o, d in zip(ori, deos):
        if o == "_________________" and d == "_________________":
            continue
        if o == "-----------------" and d == "-----------------":
            continue
        if o[0:5] == 'label' and d[0:5] == 'label':
            continue
        if o[0:5] == 'Wrong' and d[0:5] == 'Right':
            print(f"ori:{o.strip()} || deos:{d.strip()}")
            cnt += 1
    print(f"total numbers:{cnt}")



ori_lines = read_txt("res_ori.txt")
deos_lines = read_txt("deos.txt")
ana_wrong(ori_lines, deos_lines)