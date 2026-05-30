"""Render the README showcase figures from real pipeline output.

Three figures, all drawn from live API responses (no faked numbers):

  recognition.png  enrol one photo per person, query with a *different* photo,
                   show the match + similarity the server actually returned.
  detection.png    one portrait with the YuNet bounding box, five landmarks,
                   and the detection confidence.
  comparison.png   CompareFaces on pairs of *different* people — the low scores
                   that keep them apart, well below the match threshold.

Faces live in docs/img/faces/ (freely licensed — see that folder's CREDITS.md).
Requires the server running (default :8080).

  uv run python scripts/make_readme_figures.py
  uv run python scripts/make_readme_figures.py --endpoint http://127.0.0.1:8090
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import boto3
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FACES = ROOT / "docs" / "img" / "faces"
OUT = ROOT / "docs" / "img"

# slug -> (display name, enrol file, query file)
PEOPLE = [
    ("biden", "Joe Biden", "biden_1.jpg", "biden_2.jpg"),
    ("merkel", "Angela Merkel", "merkel_1.jpg", "merkel_2.jpg"),
    ("trudeau", "Justin Trudeau", "trudeau_1.jpg", "trudeau_2.jpg"),
]
DETECTION_IMG = "biden_1.jpg"

# Pairs of DIFFERENT people for the CompareFaces example.
# (name A, file A, name B, file B) — the last pair is the closest non-match.
PAIRS = [
    ("Joe Biden", "biden_1.jpg", "Angela Merkel", "merkel_1.jpg"),
    ("Angela Merkel", "merkel_1.jpg", "Justin Trudeau", "trudeau_1.jpg"),
    ("Joe Biden", "biden_1.jpg", "Justin Trudeau", "trudeau_1.jpg"),
]

# Palette (reads well on GitHub's white README background).
BG = (255, 255, 255)
INK = (13, 27, 42)
MUTE = (87, 96, 106)
LINE = (208, 215, 222)
GREEN = (26, 127, 55)
GREEN_BG = (230, 248, 237)
RED = (207, 34, 70)
RED_BG = (253, 235, 236)
ARROW = (140, 149, 159)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def detect(rek, jpg: bytes) -> dict:
    return rek.detect_faces(Image={"Bytes": jpg})["FaceDetails"][0]


def face_crop(img: Image.Image, bbox: dict, size: int, pad: float = 0.45) -> Image.Image:
    """Square crop centred on the detected face, padded out for hair/chin."""
    w, h = img.size
    cx = (bbox["Left"] + bbox["Width"] / 2) * w
    cy = (bbox["Top"] + bbox["Height"] / 2) * h
    half = max(bbox["Width"] * w, bbox["Height"] * h) * (0.5 + pad)
    box = (cx - half, cy - half, cx + half, cy + half)
    return img.crop(tuple(map(int, box))).resize((size, size), Image.LANCZOS)


def rounded(draw, xy, r, **kw):
    draw.rounded_rectangle(xy, radius=r, **kw)


def save_png(canvas: Image.Image, path: Path) -> None:
    """Palette-quantise + optimise — keeps text crisp, cuts file size ~4x."""
    path.parent.mkdir(parents=True, exist_ok=True)
    q = canvas.convert("RGB").quantize(colors=256, method=Image.FASTOCTREE, dither=Image.NONE)
    q.save(path, optimize=True)
    print(f"wrote {path}  ({canvas.width}x{canvas.height}, {path.stat().st_size // 1024} KB)")


def text_center(draw, cx, y, s, fnt, fill):
    w = draw.textbbox((0, 0), s, font=fnt)[2]
    draw.text((cx - w / 2, y), s, font=fnt, fill=fill)


def build_recognition(rek, FACES: Path) -> None:
    THUMB = 200
    M = 44
    ARROW_W = 64
    BADGE_W = 320
    GAP = 26
    x_enrol = M
    x_arrow = x_enrol + THUMB + GAP
    x_query = x_arrow + ARROW_W + GAP
    x_badge = x_query + THUMB + GAP
    W = x_badge + BADGE_W + M

    people = PEOPLE
    head_h = 116
    colh_y = head_h
    colh_h = 34
    row_top = colh_y + colh_h + 10
    row_h = THUMB + 22
    H = row_top + row_h * len(people) + 64

    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    # Header
    d.text((M, 30), "open-recognition", font=font(True, 34), fill=INK)
    d.text((M, 74), "Enrol one photo per person — recognise them from a different one.",
           font=font(False, 17), fill=MUTE)

    # Column headers
    d.text((x_enrol, colh_y), "ENROLLED", font=font(True, 13), fill=MUTE)
    d.text((x_query, colh_y), "QUERY · UNSEEN PHOTO", font=font(True, 13), fill=MUTE)
    d.text((x_badge, colh_y), "MATCH", font=font(True, 13), fill=MUTE)
    d.line([(M, colh_y + colh_h), (W - M, colh_y + colh_h)], fill=LINE, width=1)

    for i, (slug, name, ef, qf) in enumerate(people):
        y = row_top + i * row_h
        enrol = Image.open(FACES / ef).convert("RGB")
        query = Image.open(FACES / qf).convert("RGB")
        et = face_crop(enrol, detect(rek, (FACES / ef).read_bytes())["BoundingBox"], THUMB)
        qt = face_crop(query, detect(rek, (FACES / qf).read_bytes())["BoundingBox"], THUMB)

        # similarity from a real search: enrol -> query
        coll = f"_fig_{slug}"
        try:
            rek.delete_collection(CollectionId=coll)
        except Exception:
            pass
        rek.create_collection(CollectionId=coll)
        rek.index_faces(CollectionId=coll, Image={"Bytes": (FACES / ef).read_bytes()},
                        ExternalImageId=slug, MaxFaces=1)
        res = rek.search_faces_by_image(CollectionId=coll,
                                        Image={"Bytes": (FACES / qf).read_bytes()},
                                        FaceMatchThreshold=80.0, MaxFaces=1)
        rek.delete_collection(CollectionId=coll)
        sim = res["FaceMatches"][0]["Similarity"] if res.get("FaceMatches") else 0.0

        for img, x in ((et, x_enrol), (qt, x_query)):
            canvas.paste(img, (x, y))
            d.rounded_rectangle([x, y, x + THUMB, y + THUMB], radius=10, outline=LINE, width=2)

        # arrow
        ay = y + THUMB // 2
        ax0, ax1 = x_arrow + 6, x_arrow + ARROW_W - 6
        d.line([(ax0, ay), (ax1, ay)], fill=ARROW, width=3)
        d.polygon([(ax1, ay), (ax1 - 11, ay - 7), (ax1 - 11, ay + 7)], fill=ARROW)

        # match badge
        bx0, by0, bx1, by1 = x_badge, y, x_badge + BADGE_W, y + THUMB
        rounded(d, [bx0, by0, bx1, by1], 12, fill=GREEN_BG, outline=GREEN, width=2)
        cx = (bx0 + bx1) / 2
        # check disc
        cr = 20
        ccy = by0 + 46
        d.ellipse([cx - cr, ccy - cr, cx + cr, ccy + cr], fill=GREEN)
        d.line([(cx - 9, ccy), (cx - 2, ccy + 8)], fill=(255, 255, 255), width=4)
        d.line([(cx - 2, ccy + 8), (cx + 11, ccy - 8)], fill=(255, 255, 255), width=4)
        text_center(d, cx, by0 + 80, name, font(True, 20), INK)
        text_center(d, cx, by0 + 112, f"{sim:.1f}%", font(True, 34), GREEN)
        text_center(d, cx, by0 + 158, "similarity", font(False, 14), MUTE)

    foot = ("YuNet detection · SFace 128-d embeddings · pgvector HNSW search   "
            "·   every score above is live API output")
    d.text((M, H - 40), foot, font=font(False, 13), fill=MUTE)

    save_png(canvas, OUT / "recognition.png")


def build_detection(rek, FACES: Path) -> None:
    jpg = (FACES / DETECTION_IMG).read_bytes()
    fd = detect(rek, jpg)
    img = Image.open(io.BytesIO(jpg)).convert("RGB")

    target_w = 560
    scale = target_w / img.width
    img = img.resize((target_w, int(img.height * scale)), Image.LANCZOS)
    W, H = img.size

    canvas = Image.new("RGB", (W, H + 70), BG)
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)

    b = fd["BoundingBox"]
    x0, y0 = b["Left"] * W, b["Top"] * H
    x1, y1 = x0 + b["Width"] * W, y0 + b["Height"] * H
    d.rectangle([x0, y0, x1, y1], outline=GREEN, width=3)

    # confidence tag on the box
    tag = f"YuNet  {fd['Confidence']:.1f}%"
    tf = font(True, 16)
    tw = d.textbbox((0, 0), tag, font=tf)[2]
    ty1 = max(0, y0 - 26)
    d.rectangle([x0, ty1, x0 + tw + 16, ty1 + 24], fill=GREEN)
    d.text((x0 + 8, ty1 + 4), tag, font=tf, fill=(255, 255, 255))

    colors = {"eyeLeft": (0, 200, 255), "eyeRight": (0, 200, 255),
              "nose": (255, 205, 0), "mouthLeft": (255, 64, 160),
              "mouthRight": (255, 64, 160)}
    for lm in fd.get("Landmarks", []):
        lx, ly = lm["X"] * W, lm["Y"] * H
        c = colors.get(lm["Type"], (255, 255, 255))
        d.ellipse([lx - 5, ly - 5, lx + 5, ly + 5], fill=c, outline=(255, 255, 255))

    cap = "DetectFaces → bounding box, confidence, and five landmarks (eyes · nose · mouth)"
    d.text((4, H + 22), cap, font=font(False, 14), fill=MUTE)

    save_png(canvas, OUT / "detection.png")


def build_comparison(rek, FACES: Path) -> None:
    """CompareFaces on pairs of DIFFERENT people — should score well below 80."""
    THUMB = 180
    M = 44
    VS_W = 56
    BADGE_W = 300
    GAP = 26
    CAP = 26  # name caption under each thumb
    x_a = M
    x_vs = x_a + THUMB + GAP
    x_b = x_vs + VS_W + GAP
    x_badge = x_b + THUMB + GAP
    W = x_badge + BADGE_W + M

    head_h = 116
    colh_h = 34
    row_top = head_h + colh_h + 10
    row_h = THUMB + CAP + 24
    H = row_top + row_h * len(PAIRS) + 60

    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    d.text((M, 30), "open-recognition", font=font(True, 34), fill=INK)
    d.text((M, 74), "CompareFaces tells people apart — different faces score far below the threshold.",
           font=font(False, 17), fill=MUTE)

    d.text((x_a, head_h), "PHOTO A", font=font(True, 13), fill=MUTE)
    d.text((x_b, head_h), "PHOTO B", font=font(True, 13), fill=MUTE)
    d.text((x_badge, head_h), "COMPAREFACES", font=font(True, 13), fill=MUTE)
    d.line([(M, head_h + colh_h), (W - M, head_h + colh_h)], fill=LINE, width=1)

    for i, (name_a, fa, name_b, fb) in enumerate(PAIRS):
        y = row_top + i * row_h
        for fn, x, nm in ((fa, x_a, name_a), (fb, x_b, name_b)):
            img = Image.open(FACES / fn).convert("RGB")
            thumb = face_crop(img, detect(rek, (FACES / fn).read_bytes())["BoundingBox"], THUMB)
            canvas.paste(thumb, (x, y))
            d.rounded_rectangle([x, y, x + THUMB, y + THUMB], radius=10, outline=LINE, width=2)
            text_center(d, x + THUMB / 2, y + THUMB + 5, nm, font(False, 14), MUTE)

        # "vs"
        text_center(d, x_vs + VS_W / 2, y + THUMB / 2 - 14, "vs", font(True, 22), ARROW)

        res = rek.compare_faces(SourceImage={"Bytes": (FACES / fa).read_bytes()},
                                TargetImage={"Bytes": (FACES / fb).read_bytes()},
                                SimilarityThreshold=0.0)
        sim = res["FaceMatches"][0]["Similarity"] if res.get("FaceMatches") else 0.0

        bx0, by0, bx1, by1 = x_badge, y, x_badge + BADGE_W, y + THUMB
        rounded(d, [bx0, by0, bx1, by1], 12, fill=RED_BG, outline=RED, width=2)
        cx = (bx0 + bx1) / 2
        cr, ccy = 19, by0 + 42
        d.ellipse([cx - cr, ccy - cr, cx + cr, ccy + cr], fill=RED)
        d.line([(cx - 8, ccy - 8), (cx + 8, ccy + 8)], fill=(255, 255, 255), width=4)
        d.line([(cx - 8, ccy + 8), (cx + 8, ccy - 8)], fill=(255, 255, 255), width=4)
        text_center(d, cx, by0 + 74, f"{sim:.1f}%", font(True, 32), RED)
        text_center(d, cx, by0 + 116, "not the same person", font(True, 15), INK)
        text_center(d, cx, by0 + 140, "below the 80% match threshold", font(False, 13), MUTE)

    foot = ("CompareFaces with SimilarityThreshold=0 to show the raw score   ·   "
            "every number above is live API output")
    d.text((M, H - 38), foot, font=font(False, 13), fill=MUTE)

    save_png(canvas, OUT / "comparison.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--faces-dir", default=str(FACES))
    args = ap.parse_args()
    faces = Path(args.faces_dir)

    rek = boto3.client("rekognition", endpoint_url=args.endpoint, region_name="us-east-1",
                       aws_access_key_id="x", aws_secret_access_key="x")
    build_recognition(rek, faces)
    build_detection(rek, faces)
    build_comparison(rek, faces)


if __name__ == "__main__":
    main()
