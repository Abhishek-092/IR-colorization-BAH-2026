import os
import json
import glob
import hashlib
import shutil
import logging

logger = logging.getLogger(__name__)

STATE_FILE_PATH = os.path.join("output", "pipeline_state.json")

def compute_file_hash(filepath):
    """Computes SHA256 of a code/config file for stage signature tracking."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()[:16]

def get_dataset_pipeline_signature():
    """Generates a signature for dataset preparation logic & config."""
    sig_str = compute_file_hash("configs/data.yaml") + compute_file_hash("data_pipeline/prepare_dataset.py")
    return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

def get_stage1_pipeline_signature():
    """Generates a signature for Stage 1 training logic & config."""
    sig_str = (
        compute_file_hash("configs/training.yaml") +
        compute_file_hash("training/backbone.py") +
        compute_file_hash("training/sr_head.py") +
        compute_file_hash("training/trainer.py")
    )
    return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

def get_stage2_pipeline_signature():
    """Generates a signature for Stage 2 training logic & config."""
    sig_str = (
        compute_file_hash("configs/training.yaml") +
        compute_file_hash("training/mixture_head.py") +
        compute_file_hash("training/trainer.py")
    )
    return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

def get_product_input_signature(product_dir):
    """
    Creates a lightweight input signature for a Landsat product directory
    using band filenames, file sizes, and modification timestamps (no heavy file reads).
    """
    b2_files = sorted(glob.glob(os.path.join(product_dir, "*_B2.TIF")) + glob.glob(os.path.join(product_dir, "*_B2.tif")) + glob.glob(os.path.join(product_dir, "*_SR_B2.TIF")) + glob.glob(os.path.join(product_dir, "*_SR_B2.tif")))
    b3_files = sorted(glob.glob(os.path.join(product_dir, "*_B3.TIF")) + glob.glob(os.path.join(product_dir, "*_B3.tif")) + glob.glob(os.path.join(product_dir, "*_SR_B3.TIF")) + glob.glob(os.path.join(product_dir, "*_SR_B3.tif")))
    b4_files = sorted(glob.glob(os.path.join(product_dir, "*_B4.TIF")) + glob.glob(os.path.join(product_dir, "*_B4.tif")) + glob.glob(os.path.join(product_dir, "*_SR_B4.TIF")) + glob.glob(os.path.join(product_dir, "*_SR_B4.tif")))
    b10_files = sorted(glob.glob(os.path.join(product_dir, "*_B10.TIF")) + glob.glob(os.path.join(product_dir, "*_B10.tif")) + glob.glob(os.path.join(product_dir, "*_ST_B10.TIF")) + glob.glob(os.path.join(product_dir, "*_ST_B10.tif")))

    files = list(set(b2_files + b3_files + b4_files + b10_files))
    if not files:
        return ""

    meta_tokens = []
    for fpath in sorted(files):
        try:
            stat = os.stat(fpath)
            meta_tokens.append(f"{os.path.basename(fpath)}:{stat.st_size}:{stat.st_mtime}")
        except Exception:
            pass

    joined = "|".join(meta_tokens)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]

class PipelineState:
    def __init__(self, state_path=STATE_FILE_PATH):
        self.state_path = state_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse pipeline state file {self.state_path}: {e}")
        return {
            "version": "1.0",
            "products": {},
            "stage1": {},
            "stage2": {}
        }

    def save(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def purge_orphaned_products(self, valid_product_ids, patches_dir):
        """
        Removes patch directories and index records for products no longer present in input_dir.
        """
        valid_set = set(valid_product_ids)
        changed = False

        recorded_ids = list(self.data.get("products", {}).keys())
        for pid in recorded_ids:
            if pid not in valid_set:
                logger.info(f"Purging index state for deleted product: {pid}")
                del self.data["products"][pid]
                changed = True

        if os.path.exists(patches_dir):
            for item in os.listdir(patches_dir):
                item_path = os.path.join(patches_dir, item)
                if os.path.isdir(item_path) and item not in valid_set:
                    logger.info(f"Purging orphaned patch directory from disk: {item_path}")
                    shutil.rmtree(item_path)
                    changed = True

        if changed:
            self.save()

    def is_product_dataset_valid(self, product_dir, patches_dir):
        product_id = os.path.basename(product_dir.rstrip('/\\'))
        if product_id not in self.data["products"]:
            return False

        prod_state = self.data["products"][product_id]
        current_input_sig = get_product_input_signature(product_dir)
        current_ds_sig = get_dataset_pipeline_signature()

        if prod_state.get("input_signature") != current_input_sig:
            return False
        if prod_state.get("dataset_pipeline_signature") != current_ds_sig:
            return False

        target_dir = os.path.join(patches_dir, product_id)
        if not os.path.exists(target_dir):
            return False

        patch_dirs = glob.glob(os.path.join(target_dir, f"{product_id}_patch_*"))
        expected_count = prod_state.get("patch_count", 0)
        if expected_count <= 0 or len(patch_dirs) != expected_count:
            return False

        for pdir in patch_dirs[:3]:
            p_name = os.path.basename(pdir)
            f1 = os.path.join(pdir, f"{p_name}_rgb_100m.npy")
            f2 = os.path.join(pdir, f"{p_name}_tir_100m.npy")
            f3 = os.path.join(pdir, f"{p_name}_tir_200m.npy")
            if not (os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3)):
                return False
            if os.path.getsize(f1) == 0 or os.path.getsize(f2) == 0 or os.path.getsize(f3) == 0:
                return False

        return True

    def record_product_dataset(self, product_dir, patch_count):
        product_id = os.path.basename(product_dir.rstrip('/\\'))
        self.data["products"][product_id] = {
            "input_signature": get_product_input_signature(product_dir),
            "dataset_pipeline_signature": get_dataset_pipeline_signature(),
            "patch_count": patch_count
        }
        self.save()

    def get_dataset_hash(self):
        """Hashes the current dataset state across all recorded products."""
        prod_sigs = [f"{pid}:{info['input_signature']}" for pid, info in sorted(self.data["products"].items())]
        return hashlib.sha256("|".join(prod_sigs).encode()).hexdigest()[:16]

    def is_stage1_valid(self, checkpoint_dir):
        st = self.data.get("stage1", {})
        if not st.get("completed", False):
            return False

        if st.get("stage1_pipeline_signature") != get_stage1_pipeline_signature():
            return False
        if st.get("dataset_state_hash") != self.get_dataset_hash():
            return False

        bb_ckpt = os.path.join(checkpoint_dir, "backbone_stage1.pth")
        sr_ckpt = os.path.join(checkpoint_dir, "sr_head_stage1.pth")
        if not (os.path.exists(bb_ckpt) and os.path.exists(sr_ckpt)):
            return False
        if os.path.getsize(bb_ckpt) == 0 or os.path.getsize(sr_ckpt) == 0:
            return False

        return True

    def record_stage1_completion(self, checkpoint_dir):
        self.data["stage1"] = {
            "completed": True,
            "stage1_pipeline_signature": get_stage1_pipeline_signature(),
            "dataset_state_hash": self.get_dataset_hash(),
            "checkpoint_dir": checkpoint_dir
        }
        self.data["stage2"] = {}
        self.save()

    def is_stage2_valid(self, checkpoint_dir):
        st = self.data.get("stage2", {})
        if not st.get("completed", False):
            return False

        if st.get("stage2_pipeline_signature") != get_stage2_pipeline_signature():
            return False
        if st.get("dataset_state_hash") != self.get_dataset_hash():
            return False
        
        st1_hash = hashlib.sha256(json.dumps(self.data.get("stage1", {})).encode()).hexdigest()[:16]
        if st.get("stage1_state_hash") != st1_hash:
            return False

        mix_ckpt = os.path.join(checkpoint_dir, "mixture_head_stage2.pth")
        if not os.path.exists(mix_ckpt) or os.path.getsize(mix_ckpt) == 0:
            return False

        return True

    def record_stage2_completion(self, checkpoint_dir):
        st1_hash = hashlib.sha256(json.dumps(self.data.get("stage1", {})).encode()).hexdigest()[:16]
        self.data["stage2"] = {
            "completed": True,
            "stage2_pipeline_signature": get_stage2_pipeline_signature(),
            "dataset_state_hash": self.get_dataset_hash(),
            "stage1_state_hash": st1_hash,
            "checkpoint_dir": checkpoint_dir
        }
        self.save()
