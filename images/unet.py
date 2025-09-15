# unet.py  (batch mask exporter)
import os, json, argparse, cv2, numpy as np
from pathlib import Path

HIP_CATEGORY_NAMES = {"Grade_1","Grade_2","Grade_3","grade_4","Grade_5","Grade_6"}  # adjust if needed

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def polygons_to_mask(polys, h, w):
    mask = np.zeros((h, w), np.uint8)
    for seg in polys:
        if not isinstance(seg, list) or len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        pts[:,0] = np.clip(pts[:,0], 0, w-1)
        pts[:,1] = np.clip(pts[:,1], 0, h-1)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def load_coco(coco_path):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cats_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    hip_ids = {cid for cid, name in cats_by_id.items() if name in HIP_CATEGORY_NAMES}
    images = coco["images"]
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    images_by_name = {im["file_name"]: im for im in images}
    images_by_id = {im["id"]: im for im in images}
    return hip_ids, images_by_name, images_by_id, anns_by_img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="folder containing images (e.g., images/train)")
    ap.add_argument("--coco", required=True, help="path to _annotations.coco.json in the same folder")
    ap.add_argument("--out_dir", default="", help="optional folder for masks; default saves next to originals")
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    coco_path = Path(args.coco)
    out_dir = Path(args.out_dir) if args.out_dir else img_dir

    hip_ids, images_by_name, images_by_id, anns_by_img = load_coco(str(coco_path))

    # map filenames in folder to COCO image entries
    files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.name != "_annotations.coco.json"]
    files = sorted(files, key=lambda p: p.name)

    os.makedirs(out_dir, exist_ok=True)
    written = 0; skipped = 0; missing = 0

    for img_path in files:
        # find corresponding COCO image meta by exact file_name or by stem fallback
        im_meta = images_by_name.get(img_path.name)
        if im_meta is None:
            stem = img_path.stem
            for fn, im in images_by_name.items():
                if Path(fn).stem == stem:
                    im_meta = im
                    break
        if im_meta is None:
            print(f"[MISS] not found in COCO: {img_path.name}")
            missing += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[SKIP] cannot read: {img_path}")
            skipped += 1
            continue

        Hc, Wc = int(im_meta["height"]), int(im_meta["width"])
        anns = [a for a in anns_by_img.get(im_meta["id"], []) if a["category_id"] in hip_ids]
        if len(anns) == 0:
            print(f"[SKIP] no hip annotations: {img_path.name}")
            skipped += 1
            continue

        mask_coco = np.zeros((Hc, Wc), np.uint8)
        for a in anns:
            seg = a.get("segmentation", [])
            if isinstance(seg, list) and len(seg) > 0:
                mask_coco = np.maximum(mask_coco, polygons_to_mask(seg, Hc, Wc))
            else:
                # If RLE appears, consider pycocotools.coco.COCO.annToMask for decoding.
                # For now, skip non-polygon entries.
                pass

        # Resize to the actual image size (handles any mismatch)
        h0, w0 = img.shape[:2]
        mask = cv2.resize(mask_coco, (w0, h0), interpolation=cv2.INTER_NEAREST)

        mask_path = out_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"[OK] {img_path.name} -> {mask_path.name}")
        written += 1

    print(f"\nDone. masks written: {written}, skipped: {skipped}, not-in-coco: {missing}")

if __name__ == "__main__":
    main()
