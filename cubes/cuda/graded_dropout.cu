__global__ void graded_dropout_fwd_bwd(float *in_tensor, int a, int b, int u, int batch_size, 
        int channel_size, int hid_size) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    int hid = blockIdx.y * blockDim.y + threadIdx.y;
    if (bid >= batch_size || hid >= hid_size)
        return;
    float p_hat = 0;
    float dp = 1.0 / (b - a);
    for (int cid = u; cid < channel_size; ++cid) {
        int idx = bid * channel_size * hid_size + cid * hid_size + hid;
        if (p_hat < 1 - dp)
            p_hat += dp;
        in_tensor[idx] *= 1 / (1 - p_hat);
    }
}
