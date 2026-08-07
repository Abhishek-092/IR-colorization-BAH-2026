# ==========================================================================
# SUTRAM v2 — Kaggle GPU training (unified pipeline, pre-made patches)
# ==========================================================================
# Attach dataset "sutram-v2-bundle" (code.zip + patches_compact.zip).
# Accelerator = GPU, Internet = On, Run All.
# Trains Stage 1 (improved SR head) + Stage 2 (colour mixture) on the 37-scene
# DATA RELIC patches, then packages sutram_final.pth into /kaggle/working.
# Self-logs to /kaggle/working/run.log; heavy work stays in /kaggle/tmp.
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
    # 1. Get code + patches. Kaggle auto-extracts dataset zips, so each may arrive
    #    either as a .zip OR as an already-extracted tree — handle both.
    ROOT = "/kaggle/tmp/sutram"
    shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT, exist_ok=True)
    code_zip = glob.glob("/kaggle/input/**/code.zip", recursive=True)
    if code_zip:
        log("unzip code:", code_zip[0]); zipfile.ZipFile(code_zip[0]).extractall(ROOT)
    else:
        cli = glob.glob("/kaggle/input/**/cli.py", recursive=True)
        assert cli, f"no code.zip and no cli.py; input has {glob.glob('/kaggle/input/*')}"
        log("code arrived extracted:", os.path.dirname(cli[0]))
        shutil.copytree(os.path.dirname(cli[0]), ROOT, dirs_exist_ok=True)
    assert os.path.exists(os.path.join(ROOT, "cli.py")), "cli.py missing after code stage"

    def find_patch_dir(base):
        cand = glob.glob(os.path.join(base, "**", "LC0*_*"), recursive=True)
        cand = [c for c in cand if os.path.isdir(c) and glob.glob(os.path.join(c, "sample_*"))]
        return os.path.dirname(cand[0]) if cand else None

    PATCH_DIR = find_patch_dir("/kaggle/input")   # already-extracted form
    if PATCH_DIR is None:
        PATCH_ROOT = "/kaggle/tmp/patches"
        shutil.rmtree(PATCH_ROOT, ignore_errors=True); os.makedirs(PATCH_ROOT, exist_ok=True)
        pz = glob.glob("/kaggle/input/**/patches_compact.zip", recursive=True)
        assert pz, f"no patches_compact.zip and no extracted scenes; input={glob.glob('/kaggle/input/**/*', recursive=True)[:20]}"
        log("unzip patches:", pz[0]); zipfile.ZipFile(pz[0]).extractall(PATCH_ROOT)
        PATCH_DIR = find_patch_dir(PATCH_ROOT)
    assert PATCH_DIR, "no scene patch folders found"
    log("patch dir:", PATCH_DIR, "| scenes:", len(glob.glob(os.path.join(PATCH_DIR, "LC0*_*"))))

    os.chdir(ROOT); sys.path.insert(0, ROOT)

    # 2. Deps (torch is preinstalled on Kaggle; the rest are light)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "omegaconf", "opencv-python-headless", "scipy",
                    "rasterio", "tifffile", "imagecodecs"], check=False)
    import torch
    log("CUDA:", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")

    # 2b. CUDA-kernel sanity: stock Kaggle torch occasionally lacks a compiled
    #     kernel for the assigned GPU. Detect + reinstall a matching wheel.
    def cuda_ok():
        r = subprocess.run([sys.executable, "-c",
            "import torch;torch.zeros(1).cuda();"
            "(torch.zeros(2,2,device='cuda')@torch.zeros(2,2,device='cuda'))"],
            capture_output=True, text=True)
        return r.returncode == 0, r.stderr[-400:]
    ok, err = cuda_ok()
    if not ok:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        tag = "cu124" if any(v in smi for v in ("CUDA Version: 12.4","CUDA Version: 12.5","CUDA Version: 12.6")) else "cu121"
        log("CUDA kernels broken, reinstalling torch", tag, "| err:", err)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                        "torch", "torchvision", "--index-url",
                        f"https://download.pytorch.org/whl/{tag}"], check=False)
        ok, err = cuda_ok()
    if not ok:
        raise RuntimeError("CUDA unusable even after reinstall: " + err)

    # 3. Point config at the extracted patches; device=cuda; GPU batch sizes.
    import re, yaml
    b = open("configs/base_config.yaml").read()
    b = re.sub(r'device:\s*"?\w+"?', 'device: "cuda"', b)
    open("configs/base_config.yaml", "w").write(b)

    dcfg = yaml.safe_load(open("configs/data.yaml"))
    dcfg["patches_dir"] = PATCH_DIR
    yaml.safe_dump(dcfg, open("configs/data.yaml", "w"), sort_keys=False)

    tc = yaml.safe_load(open("configs/training.yaml"))
    tc["stage1"]["batch_size"] = 8   # SR head, 512-sq output
    tc["stage2"]["batch_size"] = 4   # 512-sq K6 mixture NLL is memory-heavy
    yaml.safe_dump(tc, open("configs/training.yaml", "w"), sort_keys=False)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log("config set | patches_dir:", PATCH_DIR,
        "| stage1 epochs:", tc["stage1"]["epochs"], "stage2 epochs:", tc["stage2"]["epochs"])

    # 4. Train, staging checkpoints after each stage so a later failure never
    #    discards earlier work.
    OUT = os.path.join(WORK, "sutram_trained"); os.makedirs(OUT, exist_ok=True)
    def stage_ckpts(tag):
        for f in (glob.glob("experiments/*/checkpoints/*.pth")
                  + glob.glob("experiments/*/metrics.json")
                  + glob.glob("checkpoints/*.pth")):
            shutil.copy(f, OUT)
        log(f"[{tag}] staged:", sorted(os.listdir(OUT)))

    run(f"{sys.executable} cli.py train-stage1 --force"); stage_ckpts("after-stage1")
    run(f"{sys.executable} cli.py train-stage2 --force"); stage_ckpts("after-stage2")

    # 5. Package sutram_final.pth (self-contained, no dependence on the stale
    #    release script's hardcoded experiment id).
    import torch, json, datetime
    src = glob.glob("experiments/*/checkpoints")[0]
    bb  = torch.load(f"{src}/backbone_stage1.pth", map_location="cpu")
    sr  = torch.load(f"{src}/sr_head_stage1.pth", map_location="cpu")
    mix = torch.load(f"{src}/mixture_head_stage2.pth", map_location="cpu")
    metrics = {}
    mj = glob.glob("experiments/*/metrics.json")
    if mj: metrics = json.load(open(mj[0]))
    final = {
        "model_name": "Project SUTRAM End-to-End Release",
        "version": "2.0.0",
        "backbone_state_dict": bb, "sr_head_state_dict": sr, "mixture_head_state_dict": mix,
        "metrics": metrics,
        "config": {"model_name": "Project SUTRAM", "version": "2.0.0",
                   "timestamp": datetime.datetime.utcnow().isoformat(),
                   "K_components": 6, "precision": "fp32",
                   "dataset": "DATA RELIC 37-scene", "sr_head": "multi-scale RCAB (0.6M)",
                   "trained_on": "kaggle-gpu"},
    }
    torch.save(final, os.path.join(OUT, "sutram_final.pth"))
    n = sum(v.numel() for d in (bb, sr, mix) for v in d.values())
    log("packaged sutram_final.pth  params:", f"{n:,}", "| metrics:", metrics)
    log("DONE.")

except Exception:
    log("FATAL:\n" + traceback.format_exc())
finally:
    _logf.close()
