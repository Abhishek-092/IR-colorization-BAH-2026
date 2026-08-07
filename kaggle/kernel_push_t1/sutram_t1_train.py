# ==========================================================================
# SUTRAM Track-1 — Kaggle GPU: STAGE-2 ONLY retrain (rebalanced colour head)
# ==========================================================================
# Attach datasets: sutram-v3-bundle (patches) + sutram-t1 (code4.zip +
# frozen v3 stage-1 checkpoints). GPU on, Internet on, Run All.
# Trains only the mixture head (with texture channels + rarity rebalancing
# + balanced sampler) conditioned on the FROZEN v3 backbone/SR.
# ==========================================================================
import os, sys, glob, shutil, subprocess, time, zipfile, traceback

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)
_logf = open(os.path.join(WORK, "run.log"), "w", buffering=1)
def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True); _logf.write(m + "\n")

def run(cmd):
    log(">>>", cmd); t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout: log(r.stdout[-2500:])
    if r.returncode and r.stderr: log("STDERR:", r.stderr[-2500:])
    log(f"<<< rc={r.returncode}  {time.time()-t0:.0f}s")
    if r.returncode:
        raise RuntimeError(f"command failed: {cmd}")

try:
    # 1. Track-1 code. Kaggle may deliver code4.zip as a zip OR auto-extract it
    #    into a tree. The Track-1 code is identified by the texture-channel
    #    mixture head, so locate cli.py that sits next to that marker (never the
    #    stale code.zip inside the patches bundle).
    ROOT = "/kaggle/tmp/sutram"
    shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT, exist_ok=True)
    c4 = glob.glob("/kaggle/input/**/code4.zip", recursive=True)
    if c4:
        log("unzip code4:", c4[0]); zipfile.ZipFile(c4[0]).extractall(ROOT)
    else:
        # find the extracted tree whose mixture_head.py has the texture channels
        marks = glob.glob("/kaggle/input/**/training/mixture_head.py", recursive=True)
        src = None
        for m in marks:
            if "_thermal_texture" in open(m).read():
                src = os.path.dirname(os.path.dirname(m)); break
        assert src and os.path.exists(os.path.join(src, "cli.py")), \
            f"Track-1 code not found; input tree={glob.glob('/kaggle/input/**/*', recursive=True)[:25]}"
        log("code4 arrived extracted:", src)
        shutil.copytree(src, ROOT, dirs_exist_ok=True)
    assert os.path.exists(os.path.join(ROOT, "cli.py"))
    assert "_thermal_texture" in open(os.path.join(ROOT, "training/mixture_head.py")).read(), \
        "staged code is stale (no texture head)"

    # 2. Patches (57-scene compact bundle; may arrive zipped or extracted)
    def find_patch_dir(base):
        cand = glob.glob(os.path.join(base, "**", "LC0*_*"), recursive=True)
        cand = [c for c in cand if os.path.isdir(c) and glob.glob(os.path.join(c, "sample_*"))]
        return os.path.dirname(cand[0]) if cand else None
    PATCH_DIR = find_patch_dir("/kaggle/input")
    if PATCH_DIR is None:
        PR = "/kaggle/tmp/patches"; shutil.rmtree(PR, ignore_errors=True); os.makedirs(PR)
        pz = glob.glob("/kaggle/input/**/patches_compact.zip", recursive=True)
        assert pz, "patches_compact.zip not found"
        log("unzip patches:", pz[0]); zipfile.ZipFile(pz[0]).extractall(PR)
        PATCH_DIR = find_patch_dir(PR)
    assert PATCH_DIR
    log("patch dir:", PATCH_DIR, "| scenes:", len(glob.glob(os.path.join(PATCH_DIR, "LC0*_*"))))

    os.chdir(ROOT); sys.path.insert(0, ROOT)

    # 3. Deps + CUDA sanity (same as previous kernels)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "omegaconf", "opencv-python-headless", "scipy",
                    "rasterio", "tifffile", "imagecodecs"], check=False)
    import torch
    log("CUDA:", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")
    def cuda_ok():
        r = subprocess.run([sys.executable, "-c",
            "import torch;torch.zeros(1).cuda();"
            "(torch.zeros(2,2,device='cuda')@torch.zeros(2,2,device='cuda'))"],
            capture_output=True, text=True)
        return r.returncode == 0, r.stderr[-400:]
    ok, err = cuda_ok()
    if not ok:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        tag = "cu124" if any(v in smi for v in ("12.4","12.5","12.6")) else "cu121"
        log("CUDA broken, reinstalling torch", tag)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                        "torch", "torchvision", "--index-url",
                        f"https://download.pytorch.org/whl/{tag}"], check=False)
        ok, err = cuda_ok()
    if not ok:
        raise RuntimeError("CUDA unusable: " + err)

    # 4. Config: cuda, patches dir, GPU batch sizes
    import re, yaml
    b = open("configs/base_config.yaml").read()
    b = re.sub(r'device:\s*"?\w+"?', 'device: "cuda"', b)
    open("configs/base_config.yaml", "w").write(b)
    exp_id = yaml.safe_load(open("configs/base_config.yaml"))["experiment_id"]
    dcfg = yaml.safe_load(open("configs/data.yaml")); dcfg["patches_dir"] = PATCH_DIR
    yaml.safe_dump(dcfg, open("configs/data.yaml", "w"), sort_keys=False)
    tc = yaml.safe_load(open("configs/training.yaml"))
    tc["stage1"]["batch_size"] = 8
    tc["stage2"]["batch_size"] = 4
    yaml.safe_dump(tc, open("configs/training.yaml", "w"), sort_keys=False)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 5. Frozen Stage-1: copy the v3 backbone + SR head where Stage 2 loads them
    ckdir = f"experiments/{exp_id}/checkpoints"; os.makedirs(ckdir, exist_ok=True)
    s1b = glob.glob("/kaggle/input/**/backbone_stage1.pth", recursive=True)
    s1s = glob.glob("/kaggle/input/**/sr_head_stage1.pth", recursive=True)
    assert s1b and s1s, "v3 stage-1 checkpoints not attached"
    shutil.copy(s1b[0], f"{ckdir}/backbone_stage1.pth")
    shutil.copy(s1s[0], f"{ckdir}/sr_head_stage1.pth")
    log("stage-1 (frozen, v3) staged from:", s1b[0])

    # 6. Train STAGE 2 ONLY
    OUT = os.path.join(WORK, "sutram_trained"); os.makedirs(OUT, exist_ok=True)
    run(f"{sys.executable} cli.py train-stage2 --force")
    for f in glob.glob(f"{ckdir}/*.pth"):
        shutil.copy(f, OUT)
    log("staged:", sorted(os.listdir(OUT)))
    log("DONE.")

except Exception:
    log("FATAL:\n" + traceback.format_exc())
finally:
    _logf.close()
