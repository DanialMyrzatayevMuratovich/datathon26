"""RT-DETRv2 speed variants on one video and whether the people boxes stay the same.
    uv run python -m vision.experiments.rtdetr_speed X-14
"""

import sys
import time

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from vision import people
from vision.evaluate import video_path
from vision.frames import sample


def run(frames, proc, model, dtype, batch=8):
    out, t_pre, t_net = [], 0.0, 0.0
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        t = time.perf_counter()
        inputs = proc(images=chunk, return_tensors="pt").to(people.device())
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        t_pre += time.perf_counter() - t
        t = time.perf_counter()
        with torch.no_grad():
            res = model(**inputs)
        res.logits, res.pred_boxes = res.logits.float(), res.pred_boxes.float()
        sizes = torch.tensor([(im.height, im.width) for im in chunk])
        for r in proc.post_process_object_detection(res, target_sizes=sizes, threshold=0.45):
            out.append([(round(s, 3), [round(v, 1) for v in b]) for s, lab, b in
                        zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist(), strict=True)
                        if model.config.id2label[int(lab)] == "person"])
        t_net += time.perf_counter() - t
    return out, t_pre, t_net


def diff(a, b):
    worst_box, worst_score, missing = 0.0, 0.0, 0
    for x, y in zip(a, b, strict=True):
        xs = [p for p in x if p[0] >= 0.5]
        ys = [p for p in y if p[0] >= 0.5]
        if len(xs) != len(ys):
            missing += 1
        for s, bx in xs:
            best = min(ys, key=lambda q: sum(abs(u - v) for u, v in zip(q[1], bx, strict=True)), default=None)
            if best:
                worst_box = max(worst_box, max(abs(u - v) for u, v in zip(best[1], bx, strict=True)))
                worst_score = max(worst_score, abs(best[0] - s))
    return worst_box, worst_score, missing


def main():
    with sample(video_path(sys.argv[1] if len(sys.argv) > 1 else "X-14"), fps=2.0) as fr:
        frames = list(fr.images())
    model = AutoModelForObjectDetection.from_pretrained(people.RTDETR).eval().to(people.device())
    slow = AutoImageProcessor.from_pretrained(people.RTDETR)
    fast = AutoImageProcessor.from_pretrained(people.RTDETR, use_fast=True)
    run(frames[:8], slow, model, torch.float32)  # warm-up
    base, p0, n0 = run(frames, slow, model, torch.float32)
    print(f"A now (slow proc, fp32): pre {p0:.1f}s net {n0:.1f}s")
    b, p1, n1 = run(frames, fast, model, torch.float32)
    print(f"B fast proc, fp32: pre {p1:.1f}s net {n1:.1f}s  diff box/score/count-mismatch {diff(base, b)}")
    model16 = model.to(torch.float16)
    run(frames[:8], fast, model16, torch.float16)
    c, p2, n2 = run(frames, fast, model16, torch.float16)
    print(f"C fast proc, fp16: pre {p2:.1f}s net {n2:.1f}s  diff box/score/count-mismatch {diff(base, c)}")


if __name__ == "__main__":
    main()
