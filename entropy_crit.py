from math import log2

from gini_crit import crit


def entropy(c1_count, c2_count) -> float:
    uh_count = c1_count + c2_count
    p1 = c1_count / uh_count
    p2 = c2_count / uh_count
    return -(p1 * (log2(p1) if p1 else 0) + p2 * (log2(p2) if p2 else 0))


if __name__ == "__main__":
    classes = input().replace(',', '').split()
    for i in range(1, len(classes)):
        print(round(crit(classes[:i], classes[i:], entropy), 3), end=' ')
    print()
