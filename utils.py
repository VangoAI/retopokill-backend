def rotate(l: list, n: int):
    return l[n:] + l[:n]

def get_rotations(l: list):
    return [rotate(l, i) for i in range(len(l))]