from typing import List


def gini(c1_count, c2_count):
    uh_count = c1_count + c2_count
    return 1 - ((c1_count / uh_count) ** 2 + (c2_count / uh_count) ** 2)


def crit(u1: List[str], u2: List[str]):
    u = u1.copy()
    u.extend(u2)
    c = sorted(set(u))
    u1_c1_count = u1.count(c[0])
    u1_c2_count = u1.count(c[1])
    u2_c1_count = u2.count(c[0])
    u2_c2_count = u2.count(c[1])
    return ((len(u1) / len(u)) * gini(u1_c1_count, u1_c2_count)) + ((len(u2) / len(u)) * gini(u2_c1_count, u2_c2_count))


if __name__ == "__main__":
    classes = input().replace(',', '').split()
    for i in range(1, len(classes)):
        print(round(crit(classes[:i], classes[i:]), 3), end=' ')
    print()
