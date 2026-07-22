"""Fetch pretrained weights the cascade needs that don't auto-download.

  * SCRFD (stage ①)  -> auto-downloads via InsightFace on first run (buffalo_sc)
  * 6DRepNet (stage ②) -> auto-downloads via the `sixdrepnet` package on first run
  * eye-state (stage ③) -> fetched here: OpenVINO OMZ `open-closed-eye-0001`,
                           served directly as ONNX (32x32, 2-class).

Run once:
    python -m train.download_models
"""
from __future__ import annotations

import os
import ssl
import urllib.request

EYE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/"
    "2022.1/open-closed-eye-0001/open_closed_eye.onnx"
)
EYE_DST = os.path.join("models", "eye_state", "open_closed_eye.onnx")

# Eye gaze (stage ⑤): gaze-estimation-adas-0002, OpenVINO IR (xml + bin).
GAZE_BASE = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/"
    "models_bin/1/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002."
)
GAZE_DIR = os.path.join("models", "gaze")


def _download(url: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  already present: {dst} ({os.path.getsize(dst)} bytes)")
        return
    print(f"  downloading {url}")
    print(f"          -> {dst}")
    # Prefer requests (bundles certifi) — fixes the conda "unable to get local
    # issuer certificate" SSL error. Fall back to urllib with certifi's CA bundle.
    try:
        import requests
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    except Exception as e:   # noqa: BLE001
        print(f"  requests path failed ({e}); retrying with certifi CA bundle")
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except Exception:  # noqa: BLE001
            pass
        with urllib.request.urlopen(url, context=ctx, timeout=60) as resp, open(dst, "wb") as f:
            f.write(resp.read())
    print(f"  done ({os.path.getsize(dst)} bytes)")


def _inspect(onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed — skipping shape check)")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    i = sess.get_inputs()[0]
    o = sess.get_outputs()[0]
    print(f"  input  {i.name} {i.shape}")
    print(f"  output {o.name} {o.shape}")


def main():
    print("Eye-state model (open-closed-eye-0001):")
    _download(EYE_URL, EYE_DST)
    _inspect(EYE_DST)
    print("\nEye-gaze model (gaze-estimation-adas-0002, OpenVINO IR):")
    for ext in ("xml", "bin"):
        _download(GAZE_BASE + ext, os.path.join(GAZE_DIR, f"gaze-estimation-adas-0002.{ext}"))
    print("\nSCRFD + 6DRepNet auto-download on first pipeline run — nothing to do.")
    print("Done.")


if __name__ == "__main__":
    main()
