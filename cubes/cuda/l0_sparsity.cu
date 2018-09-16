extern "C"
__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0 / (1 + exp(-x));
}

extern "C"
__device__ __forceinline__ float training_q_fwd(const float log_alpha, const float beta, const float gamma, const float zeta) {
    return sigmoid_f(log_alpha - beta * (-gamma / zeta));
}

extern "C"
__device__ __forceinline__ float training_q_bwd(const float log_alpha, const float beta, const float gamma, const float zeta) {
    float s = training_q_fwd(log_alpha, beta, gamma, zeta);
    return s * (1 - s);
}

extern "C"
__global__ void l0_norm_fwd(float *out_norm, const float *log_alpha, const float beta, const float gamma, const float zeta, 
        const int channel_size, const int group_size) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= channel_size)
        return;
    out_norm[cid] = group_size * training_q_fwd(log_alpha[cid], beta, gamma, zeta);
}

extern "C"
__global__ void l0_norm_bwd(float *out_norm_grad, const float *in_norm_grad, const float *log_alpha, const float beta, const float gamma, const float zeta, 
        const int channel_size, const int group_size) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= channel_size)
        return;
    out_norm_grad[cid] = group_size * training_q_bwd(log_alpha[cid], beta, gamma, zeta) * in_norm_grad[cid];
}

extern "C"
__global__ void l0_weights_fwd(float *out_weights, const float *in_weights, const float *log_alpha, const float *uniform_tensor,
        const float beta, const float gamma, const float zeta, const int channel_size, const int group_size) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    int gid = blockIdx.y * blockDim.y + threadIdx.y;
    if (cid >= channel_size || gid >= group_size)
        return;
    int idx = cid * group_size + gid;
    float u = uniform_tensor[cid];
    float log_alpha_ = log_alpha[cid];
    float s = sigmoid_f((log(u / (1 - u)) + log_alpha_) / beta);
    s = s * (zeta - gamma) + gamma;
    float z = min(1.0f, max(0.0f, s));
    out_weights[idx] = z * in_weights[idx];
}

extern "C"
__global__ void l0_weights_bwd(float *out_weights_grad, float *out_log_alpha_grad, const float *in_weights_grad, const float *in_weights,
        const float *log_alpha, const float *uniform_tensor, const float beta, const float gamma, const float zeta, const int channel_size, const int group_size) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    int gid = blockIdx.y * blockDim.y + threadIdx.y;
    if (cid >= channel_size || gid >= group_size)
        return;
    float u = uniform_tensor[cid];
    int idx = cid * group_size + gid;
    float log_alpha_ = log_alpha[cid];
    float s = sigmoid_f((log(u / (1 - u)) + log_alpha_) / beta);
    float z = s * (zeta - gamma) + gamma;
    float w = in_weights[idx];
    z = min(1.0f, max(0.0f, z));
    out_weights_grad[idx] = z * in_weights_grad[idx];
    out_log_alpha_grad[idx] = w * (zeta - gamma) * s * (1 - s) / beta * in_weights_grad[idx];
}
