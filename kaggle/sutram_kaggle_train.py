# ==========================================================================
# SUTRAM — Kaggle GPU training (robust, self-logging)
# ==========================================================================
# Attach dataset "sutram-bundle", Accelerator = GPU, Internet = On, Run All.
# Writes /kaggle/working/run.log (small) + /kaggle/working/sutram_trained/*.pth.
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
    if r.stdout: log(r.stdout[-1800:])
    if r.returncode and r.stderr: log("STDERR:", r.stderr[-1800:])
    log(f"<<< rc={r.returncode}  {time.time()-t0:.0f}s")
    if r.returncode:
        raise RuntimeError(f"command failed: {cmd}")

try:
    # 1. Unpack the bundle into /kaggle/tmp (NOT /kaggle/working -> no output bloat)
    ROOT = "/kaggle/tmp/sutram"
    shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT, exist_ok=True)
    zips = (glob.glob("/kaggle/input/**/*bundle*.zip", recursive=True)
            or glob.glob("/kaggle/input/**/*.zip", recursive=True))
    if zips:
        log("unzip", zips[0]); zipfile.ZipFile(zips[0]).extractall(ROOT)
    if not os.path.exists(os.path.join(ROOT, "cli.py")):
        inner = glob.glob(os.path.join(ROOT, "*", "cli.py"))
        if inner:
            ROOT = os.path.dirname(inner[0])
    if not os.path.exists(os.path.join(ROOT, "cli.py")):
        t = glob.glob("/kaggle/input/**/cli.py", recursive=True)
        assert t, f"no cli.py; input has {glob.glob('/kaggle/input/*')}"
        shutil.copytree(os.path.dirname(t[0]), ROOT, dirs_exist_ok=True)
    # Overlay the latest CODE on top of the (older) scene bundle — the scenes
    # are large and stable; the code changes often, so it ships as a tiny
    # separate dataset (code_overlay.zip) extracted over the tree.
    # Kaggle auto-extracts uploaded zips, so the code overlay may arrive either
    # as code_overlay.zip or as an already-extracted tree. Handle both; refuse
    # to run stale code if neither is present.
    ov = glob.glob("/kaggle/input/**/code_overlay.zip", recursive=True)
    if ov:
        log("applying code overlay (zip):", ov[0])
        zipfile.ZipFile(ov[0]).extractall(ROOT)
    else:
        marker = glob.glob("/kaggle/input/*/training/material_head.py") or \
                 glob.glob("/kaggle/input/**/training/material_head.py", recursive=True)
        assert marker, ("code overlay not mounted (no zip, no extracted tree) — "
                        "refusing to silently train stale code.")
        src = os.path.dirname(os.path.dirname(marker[0]))
        log("applying code overlay (extracted tree):", src)
        shutil.copytree(src, ROOT, dirs_exist_ok=True)
    assert os.path.exists(os.path.join(ROOT, "training", "material_head.py")), \
        "overlay applied but material_head.py missing — overlay is stale"
    log("overlay verified: material_head.py present")

    os.chdir(ROOT); sys.path.insert(0, ROOT)
    log("root:", ROOT, "| scenes:", len(glob.glob("input/LC*/")))

    # 2. Dependencies (imagecodecs is required for LZW-compressed L2SP TIFs)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio",
                    "tifffile", "imagecodecs", "opencv-python-headless",
                    "omegaconf", "scipy"], check=False)
    import torch
    log("CUDA:", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-ONLY")

    # 2b. CUDA sanity check — the stock Kaggle torch sometimes lacks compiled
    #     kernels for the assigned GPU ("no kernel image available"). Detect it
    #     with a tiny op and, if broken, reinstall a torch wheel that matches the
    #     driver's CUDA version. Training runs in subprocesses, so they pick up
    #     whatever torch ends up installed.
    def cuda_ok():
        r = subprocess.run([sys.executable, "-c",
                            "import torch;torch.zeros(1).cuda();"
                            "(torch.zeros(2,2,device='cuda')@torch.zeros(2,2,device='cuda'))"],
                           capture_output=True, text=True)
        return r.returncode == 0, r.stderr[-400:]
    ok, err = cuda_ok()
    if not ok:
        log("CUDA kernels broken, reinstalling torch. err:", err)
        # pick a cuda tag from nvidia-smi (fallback cu121)
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        tag = "cu124" if "CUDA Version: 12.4" in smi or "CUDA Version: 12.5" in smi or "CUDA Version: 12.6" in smi else "cu121"
        log("reinstalling torch", tag)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--force-reinstall", "torch", "torchvision",
                        "--index-url", f"https://download.pytorch.org/whl/{tag}"], check=False)
        ok, err = cuda_ok()
        log("CUDA after reinstall:", ok, err if not ok else "")
    if not ok:
        raise RuntimeError("CUDA unusable on this GPU even after torch reinstall: " + err)

    # 3. Force device=cuda; per-stage batch sizes (the 512-sq multi-scale mixture
    #    head is memory-heavy, so Stage 2 must use a small batch to fit 16 GB).
    import re, yaml
    s = open("configs/base_config.yaml").read()
    open("configs/base_config.yaml", "w").write(re.sub(r'device:\s*"?\w+"?', 'device: "cuda"', s))
    tc = yaml.safe_load(open("configs/training.yaml"))
    # NOTE: the validation loader is built with the STAGE-1 batch size, and it is
    # reused for Stage-2 validation (which runs the heavy 512-sq mixture head). So
    # stage1 batch must also be tiny here, or Stage-2 validation OOMs. Stage 1 is
    # skipped (loaded from the dataset), so this value only governs that loader.
    tc["stage1"]["batch_size"] = 2
    tc["stage2"]["batch_size"] = 2       # 512-sq K6 mixture NLL -> tiny batch to fit 16 GB
    tc["stage1"]["epochs"] = 20
    tc["stage2"]["epochs"] = 15
    yaml.safe_dump(tc, open("configs/training.yaml", "w"), sort_keys=False)
    # Reduce CUDA fragmentation (the OOM note recommends this).
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 4. Colour patches from all input scenes
    run(f"{sys.executable} data_pipeline/prepare_dataset.py --force")

    # 5. SR pairs — generate from the FULL-RES input scenes (the 'new data/' dir
    #    isn't in the bundle, so drive prepare_sr_scenes over input/ B10 bands).
    import importlib.util
    spec = importlib.util.spec_from_file_location("psr", "scripts/prepare_sr_scenes.py")
    psr = importlib.util.module_from_spec(spec); spec.loader.exec_module(psr)
    n_sr = 0
    for d in sorted(glob.glob("input/LC*/")):
        b10 = glob.glob(f"{d}*_ST_B10.TIF") or glob.glob(f"{d}*_B10.TIF")
        if b10:
            try:
                psr.process(b10[0]); n_sr += 1
            except Exception as e:
                log("SR gen skip", d, repr(e))
    log("SR source scenes processed:", n_sr,
        "| SR patch folders:", len(glob.glob("output/patches/SR_*/")))

    # 6. Rebuild the config split from what actually got generated
    import yaml
    cfg = yaml.safe_load(open("configs/data.yaml"))
    import numpy as np
    def label_ok(scene_dir):
        # Auto-exclude label-anomalous scenes: an 'urban' fraction >25% is not a
        # real land-cover statistic — it is cloud contamination in the optical GT
        # (bright grey cloud pseudo-labels as urban) and would poison the head.
        labs = [np.load(f) for f in glob.glob(scene_dir + "/sample_*/material_512.npy")]
        if not labs:
            return True
        a = np.concatenate([l.ravel() for l in labs]); a = a[a >= 0]
        return a.size == 0 or (a == 3).mean() <= 0.25
    colour = sorted(os.path.basename(x.rstrip("/")) for x in glob.glob("output/patches/LC*/")
                    if len(glob.glob(x + "/sample_*")) >= 20 and label_ok(x.rstrip("/")))
    sr = sorted(os.path.basename(x.rstrip("/")) for x in glob.glob("output/patches/SR_*/"))
    val_c = colour[-2:] if len(colour) > 3 else colour[-1:]
    cfg["splits"]["train"] = [c for c in colour if c not in val_c]
    cfg["splits"]["val"] = val_c
    cfg["splits"]["sr_train"] = [x for x in sr if not x.endswith("_val")]
    cfg["splits"]["sr_val"] = [x for x in sr if x.endswith("_val")][:4]
    yaml.safe_dump(cfg, open("configs/data.yaml", "w"), sort_keys=False)
    log("splits — colour train/val:", len(cfg["splits"]["train"]), "/", len(cfg["splits"]["val"]),
        "| SR train/val:", len(cfg["splits"]["sr_train"]), "/", len(cfg["splits"]["sr_val"]))

    # 7. Train — stage checkpoints after EACH step so a later failure never
    #    discards earlier work (Stage 1 alone is ~1 h on a P100).
    OUT = os.path.join(WORK, "sutram_trained"); os.makedirs(OUT, exist_ok=True)
    def stage_ckpts(tag):
        for f in (glob.glob("checkpoints/*.pth") + glob.glob("experiments/*/checkpoints/*.pth")
                  + glob.glob("experiments/*/metrics.json")):
            shutil.copy(f, OUT)
        log(f"[{tag}] staged:", sorted(os.listdir(OUT)))

    # Stage 1 (SR) is already at the data ceiling (GPU retrain tied the committed
    # model), so REUSE the bundled SR checkpoints and only retrain the colour head
    # — the one thing that never fit locally. Extract stage1 weights from the
    # packaged checkpoint into the location Stage 2 loads from, then skip Stage 1.
    os.makedirs("experiments/sutram_baseline/checkpoints", exist_ok=True)
    # Prefer the attached 'sutram-stage1' dataset (checkpoints saved by the Kaggle
    # torch, so they always deserialize here). Fall back to retraining Stage 1.
    s1 = glob.glob("/kaggle/input/**/backbone_stage1.pth", recursive=True)
    s1sr = glob.glob("/kaggle/input/**/sr_head_stage1.pth", recursive=True)
    if s1 and s1sr:
        shutil.copy(s1[0], "experiments/sutram_baseline/checkpoints/backbone_stage1.pth")
        shutil.copy(s1sr[0], "experiments/sutram_baseline/checkpoints/sr_head_stage1.pth")
        log("using attached Stage-1 SR checkpoints; skipping Stage 1")
    else:
        run(f"{sys.executable} cli.py train-stage1 --force"); stage_ckpts("after-stage1")

    run(f"{sys.executable} cli.py train-stage2 --force"); stage_ckpts("after-stage2")
    run(f"{sys.executable} cli.py evaluate")
    run(f"{sys.executable} scripts/prepare_release_checkpoints.py"); stage_ckpts("final")
    log("DONE.")

except Exception:
    log("FATAL:\n" + traceback.format_exc())
finally:
    _logf.close()
