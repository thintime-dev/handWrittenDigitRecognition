# src/ocl_runtime.py
import os
import numpy as np
import pyopencl as cl

class OCLRuntime:
    def __init__(self, kernel_src_path, binary_path=None, platform_hint=None, device_hint=None):
        # 选择平台/设备
        platforms = cl.get_platforms()
        plat = None
        if platform_hint:
            for p in platforms:
                if platform_hint.lower() in p.name.lower():
                    plat = p; break
        plat = plat or platforms[0]
        devices = plat.get_devices()
        dev = None
        if device_hint:
            for d in devices:
                if device_hint.lower() in d.name.lower():
                    dev = d; break
        dev = dev or devices[0]

        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx)
        print(f"[OCL] Platform: {plat.name}")
        print(f"[OCL] Device:   {dev.name}")

        # 构建程序：优先从二进制（FPGA 位流），否则从源码编译（GPU/CPU 调试）
        if binary_path and os.path.exists(binary_path):
            with open(binary_path, "rb") as f:
                binary = f.read()
            self.prog = cl.Program(self.ctx, [dev], [binary]).build()
            print(f"[OCL] Loaded binary: {binary_path}")
        else:
            with open(kernel_src_path, "r", encoding="utf-8") as f:
                src = f.read()
            self.prog = cl.Program(self.ctx, src).build()
            print(f"[OCL] Built from source: {kernel_src_path}")

    def conv3x3_nhwc(self, inp, W, b, H, Wd, Cin, Cout):
        # inp: (H*Wd*Cin) float32
        # W: (Cout*3*3*Cin), b: (Cout)
        ctx, q = self.ctx, self.queue
        mf = cl.mem_flags

        out = np.empty((H*Wd*Cout,), dtype=np.float32)

        buf_in = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=inp)
        buf_w  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=W)
        buf_b  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=b)
        buf_out= cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)

        krn = self.prog.conv3x3_nhwc
        global_size = (Wd, H, Cout)
        krn.set_args(buf_in, buf_w, buf_b, buf_out,
                     np.int32(H), np.int32(Wd), np.int32(Cin), np.int32(Cout))
        cl.enqueue_nd_range_kernel(q, krn, global_size, None)
        cl.enqueue_copy(q, out, buf_out)
        q.finish()
        return out

    def maxpool2x2_nhwc(self, inp, H, Wd, C):
        Hout, Wout = H//2, Wd//2
        out = np.empty((Hout*Wout*C,), dtype=np.float32)

        ctx, q = self.ctx, self.queue
        mf = cl.mem_flags
        buf_in  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=inp)
        buf_out = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)

        krn = self.prog.maxpool2x2_nhwc
        global_size = (Wout, Hout, C)
        krn.set_args(buf_in, buf_out, np.int32(H), np.int32(Wd), np.int32(C))
        cl.enqueue_nd_range_kernel(q, krn, global_size, None)
        cl.enqueue_copy(q, out, buf_out)
        q.finish()
        return out

