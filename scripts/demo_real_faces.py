"""Recognise real faces through the boto3 SDK — a runnable smoke demo.

Indexes ONE photo per person, then queries with DIFFERENT, previously
unseen photos of the same people, plus a stranger who was never enrolled.
It exercises the whole stack the way a real boto3 user would: detect →
embed → index → HNSW search → calibrated similarity.

Uses LFW (Labeled Faces in the Wild). Grab it first (~243 MB, no auth):
  mkdir -p /tmp/lfw && cd /tmp/lfw
  curl -sLO https://ndownloader.figshare.com/files/5976015
  mv 5976015 lfw.tgz && tar xzf lfw.tgz

Then, with the server running on :8080:
  uv run python scripts/demo_real_faces.py --lfw /tmp/lfw/lfw_funneled

Exits 0 only if every person is recognised from their unseen photo.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import boto3

# People with several photos each, so we can index one and query another.
DEFAULT_PEOPLE = [
    "George_W_Bush",
    "Colin_Powell",
    "Tony_Blair",
    "Donald_Rumsfeld",
    "Gerhard_Schroeder",
]
DEFAULT_STRANGER = "Hugo_Chavez"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lfw", required=True, help="path to lfw_funneled directory")
    p.add_argument("--endpoint", default="http://127.0.0.1:8080")
    p.add_argument("--collection", default="real_faces_demo")
    p.add_argument("--threshold", type=float, default=80.0)
    p.add_argument("--people", nargs="+", default=DEFAULT_PEOPLE)
    p.add_argument("--stranger", default=DEFAULT_STRANGER)
    return p.parse_args()


def imgs(lfw: str, person: str) -> list[str]:
    return sorted(glob.glob(f"{lfw}/{person}/*.jpg"))


def load(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main() -> int:
    args = parse_args()
    client = boto3.client(
        "rekognition",
        endpoint_url=args.endpoint,
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="x",
    )

    # Fresh collection.
    try:
        client.delete_collection(CollectionId=args.collection)
    except client.exceptions.ResourceNotFoundException:
        pass
    client.create_collection(CollectionId=args.collection)
    print(f"Created collection '{args.collection}' at {args.endpoint}\n")

    # --- Index: first photo of each person ---
    print("=== INDEXING (1 photo per person) ===")
    for person in args.people:
        files = imgs(args.lfw, person)
        if not files:
            print(f"  {person:22s} SKIPPED (no images under {args.lfw})")
            continue
        rec = client.index_faces(
            CollectionId=args.collection,
            Image={"Bytes": load(files[0])},
            ExternalImageId=person,
            MaxFaces=1,
        )
        fr = rec["FaceRecords"]
        if fr:
            conf = fr[0]["FaceDetail"]["Confidence"]
            print(f"  {person:22s} indexed  conf={conf:5.1f}  "
                  f"{os.path.basename(files[0])}")
        else:
            print(f"  {person:22s} NOT indexed "
                  f"(unindexed: {rec.get('UnindexedFaces')})")

    desc = client.describe_collection(CollectionId=args.collection)
    print(f"\nCollection now holds {desc['FaceCount']} faces "
          f"(model {desc['FaceModelVersion']})\n")

    # --- Recognition: query with DIFFERENT, unseen photos ---
    print("=== SEARCH (different, previously-unseen photo of each person) ===")
    correct = total = 0
    for person in args.people:
        files = imgs(args.lfw, person)
        if len(files) < 2:
            continue
        query = files[1]  # second photo — never indexed
        res = client.search_faces_by_image(
            CollectionId=args.collection,
            Image={"Bytes": load(query)},
            FaceMatchThreshold=args.threshold,
            MaxFaces=3,
        )
        matches = res.get("FaceMatches", [])
        total += 1
        if matches:
            top = matches[0]
            got = top["Face"]["ExternalImageId"]
            sim = top["Similarity"]
            ok = "match" if got == person else "WRONG"
            if got == person:
                correct += 1
            print(f"  [{ok}] {person:22s} -> {got:22s} sim={sim:5.1f}")
        else:
            print(f"  [MISS] {person:22s} -> no match >= {args.threshold:.0f}")

    # --- Stranger: someone NOT in the collection ---
    print("\n=== STRANGER (not enrolled — should NOT match) ===")
    stranger_files = imgs(args.lfw, args.stranger)
    if stranger_files:
        res = client.search_faces_by_image(
            CollectionId=args.collection,
            Image={"Bytes": load(stranger_files[0])},
            FaceMatchThreshold=args.threshold,
            MaxFaces=3,
        )
        m = res.get("FaceMatches", [])
        if not m:
            print(f"  [ok] {args.stranger} -> no match (correct: not enrolled)")
        else:
            print(f"  [FALSE POSITIVE] {args.stranger} -> "
                  f"{m[0]['Face']['ExternalImageId']} sim={m[0]['Similarity']:.1f}")

    # --- CompareFaces: same vs different ---
    print("\n=== COMPARE FACES ===")
    ref = args.people[0]
    other = args.people[1]
    ref_imgs = imgs(args.lfw, ref)
    if len(ref_imgs) >= 6:
        same = client.compare_faces(
            SourceImage={"Bytes": load(ref_imgs[0])},
            TargetImage={"Bytes": load(ref_imgs[5])},
            SimilarityThreshold=0.0,
        )
        s = same["FaceMatches"][0]["Similarity"] if same["FaceMatches"] else 0
        print(f"  {ref} vs {ref} (diff photos):  {s:5.1f}%  (expect high)")
    diff = client.compare_faces(
        SourceImage={"Bytes": load(ref_imgs[0])},
        TargetImage={"Bytes": load(imgs(args.lfw, other)[0])},
        SimilarityThreshold=0.0,
    )
    d = diff["FaceMatches"][0]["Similarity"] if diff["FaceMatches"] else 0
    print(f"  {ref} vs {other} (diff people):  {d:5.1f}%  (expect low)")

    print(f"\n=== RESULT: {correct}/{total} people recognised from unseen photos ===")
    return 0 if correct == total and total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
