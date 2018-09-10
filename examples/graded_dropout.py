import cubes.torch.functional as C
import torch


def main():
    x = torch.ones(1, 40, 5).cuda()
    x.requires_grad = True
    t = C.graded_dropout(x, a=10, b=50, training=True).sum()
    t.backward()
    print(x.grad)


if __name__ == "__main__":
    main()
