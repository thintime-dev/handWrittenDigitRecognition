# src/fold_bn.py
import numpy as np, argparse

def fuse_conv(W, b, gamma, beta, mean, var, eps):
    # W: (Cout,3,3,Cin); b: (Cout,)
    scale = gamma / np.sqrt(var + eps)   # (Cout,)
    Wf = W * scale[:,None,None,None]
    bf = beta + (b - mean) * scale
    return Wf.astype(np.float32), bf.astype(np.float32)

def fuse_dense(W, b, gamma, beta, mean, var, eps):
    scale = gamma / np.sqrt(var + eps)   # (Cout,)
    Wf = W * scale[None,:]
    bf = beta + (b - mean) * scale
    return Wf.astype(np.float32), bf.astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in-npz", default="outputs/models/weights_fp32.npz")
    ap.add_argument("--out-npz", default="outputs/models/weights_fused.npz")
    args=ap.parse_args()
    z=np.load(args.in_npz)

    out={}
    # conv0..3 + bn0..3
    for i in range(4):
        W,b = z[f"conv{i}_W"], z[f"conv{i}_b"]
        g,be,mu,va,eps = z[f"bn{i}_gamma"], z[f"bn{i}_beta"], z[f"bn{i}_mean"], z[f"bn{i}_var"], float(z[f"bn{i}_eps"][0])
        Wf,bf = fuse_conv(W,b,g,be,mu,va,eps)
        out[f"conv{i}_W"]=Wf; out[f"conv{i}_b"]=bf

    # dense0 + bn4
    W0,b0 = z["dense0_W"], z["dense0_b"]
    g,be,mu,va,eps = z["bn4_gamma"], z["bn4_beta"], z["bn4_mean"], z["bn4_var"], float(z["bn4_eps"][0])
    W0f,b0f = fuse_dense(W0,b0,g,be,mu,va,eps)
    out["dense0_W"]=W0f; out["dense0_b"]=b0f

    # dense1 原样
    out["dense1_W"]=z["dense1_W"].astype(np.float32)
    out["dense1_b"]=z["dense1_b"].astype(np.float32)

    np.savez_compressed(args.out_npz, **out)
    print("[INFO] BN fused ->", args.out_npz)

if __name__=="__main__":
    main()

