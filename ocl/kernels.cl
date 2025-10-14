// ocl/kernels.cl
// 简化、可移植：float32，NHWC 输入，3x3 卷积（padding=same, stride=1），2x2 最大池化（stride=2）
// 目标：MNIST CNN（28x28，3x3 卷积、2x2池化）功能正确优先；后续可再做向量化/本地缓存优化/定点化

// 卷积：输入 NHWC: (H, W, Cin)，权重: (Cout, Kh=3, Kw=3, Cin)，偏置: (Cout)
// 输出 NHWC: (H, W, Cout)  with SAME padding
__kernel void conv3x3_nhwc(
    __global const float* input,   // [H*W*Cin]
    __global const float* weight,  // [Cout*3*3*Cin]
    __global const float* bias,    // [Cout]
    __global float* output,        // [H*W*Cout]
    int H, int W, int Cin, int Cout
){
    int x = get_global_id(0); // width
    int y = get_global_id(1); // height
    int co = get_global_id(2); // output channel

    if (x >= W || y >= H || co >= Cout) return;

    float acc = bias[co];
    // 3x3 SAME padding
    for (int ky=0; ky<3; ++ky){
        for (int kx=0; kx<3; ++kx){
            int in_y = y + ky - 1;
            int in_x = x + kx - 1;
            if (in_y < 0 || in_y >= H || in_x < 0 || in_x >= W) continue;

            int in_base = (in_y*W + in_x)*Cin;
            int w_base  = ((co*3 + ky)*3 + kx)*Cin;

            for (int ci=0; ci<Cin; ++ci){
                acc += input[in_base + ci] * weight[w_base + ci];
            }
        }
    }
    output[(y*W + x)*Cout + co] = acc;
}

// ReLU + BatchNorm（推理态）：y = relu( (x - mean)*gamma/inv_std + beta )
// 为简化：这里在 Host 侧做 BN+ReLU（NumPy），如需搬到设备可实现本核
// 这里仍提供一个 ReLU 内核（可选）
__kernel void relu_inplace(__global float* x, int N){
    int i = get_global_id(0);
    if (i < N){
        float v = x[i];
        x[i] = v > 0.0f ? v : 0.0f;
    }
}

// 2x2 最大池化，stride=2，NHWC
__kernel void maxpool2x2_nhwc(
    __global const float* input,   // [H*W*C]
    __global float* output,        // [Hout*Wout*C]
    int H, int W, int C
){
    int x = get_global_id(0); // width_out
    int y = get_global_id(1); // height_out
    int c = get_global_id(2);

    int Hout = H / 2;
    int Wout = W / 2;

    if (x >= Wout || y >= Hout || c >= C) return;

    int in_x = x * 2;
    int in_y = y * 2;

    float m = -3.4e38f; // -inf
    for (int dy=0; dy<2; ++dy){
        for (int dx=0; dx<2; ++dx){
            int ix = in_x + dx;
            int iy = in_y + dy;
            float v = input[(iy*W + ix)*C + c];
            m = v > m ? v : m;
        }
    }
    output[(y*Wout + x)*C + c] = m;
}

