import cubes
import torch
import numpy as np


def main():
    t = torch.ones(3, 4, 5).cuda()
    cubes.load("graded_dropout.cu").graded_dropout_fwd_bwd(*cubes.wrap(t), np.int32(1), np.int32(5), 
        np.int32(2), np.int32(3), np.int32(4), np.int32(5), grid=(3, 5), block=(4, 1, 1))
    torch.cuda.synchronize()
    print(t)


if __name__ == "__main__":
    main()