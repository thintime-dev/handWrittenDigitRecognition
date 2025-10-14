# tests/compare_ocl_vs_tf.py
import os, subprocess, json, numpy as np, tempfile, shutil, sys
from glob import glob

IMG_DIR = sys.argv[1] if len(sys.argv)>1 else "res/test_images"
TF_CMD = ["python","-m","src.predict","--img-dir",IMG_DIR,"--out-csv","/tmp/tf.csv"]
OCL_CMD = ["python","-m","src.predict_ocl","--img-dir",IMG_DIR,"--weights","outputs/models/weights_fp32.npz","--kernel","ocl/kernels.cl","--platform","NVIDIA"]

subprocess.run(TF_CMD, check=True)
tf = {}
with open("/tmp/tf.csv") as f:
    next(f)
    for line in f: 
        name,p,conf = line.strip().split(",")
        tf[name]=(int(p),float(conf))

out = subprocess.check_output(OCL_CMD, text=True)
ocl={}
for line in out.splitlines():
    if "->" in line and "(" in line:
        name=line.split("->")[0].strip()
        pred=line.split("->")[1].split("(")[0].strip()
        conf=line.split("conf=")[1].split(")")[0]
        ocl[name]=(int(pred),float(conf))

names=sorted(set(tf) & set(ocl))
eq = sum(tf[n][0]==ocl[n][0] for n in names)
print(f"Samples={len(names)}, Agree={eq}, Acc={(eq/len(names)):.4f}")
max_conf_gap = max(abs(tf[n][1]-ocl[n][1]) for n in names) if names else 0.0
print(f"Max confidence gap: {max_conf_gap:.6f}")

