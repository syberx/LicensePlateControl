import os
# CRITICAL CPU OPTIMIZATION FOR DOCKER:
# Prevent PyTorch/ONNX from over-spawning threads on virtualized CPUs,
# which causes massive context switching overhead and slows down inference.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
cv2.setNumThreads(2)
import numpy as np
import base64
import time
import traceback

VERSION = "1.2.0"

alpr = None
alpr_error = None
_ov_device = os.environ.get("ORT_OPENVINO_DEVICE", "CPU")

def init_alpr():
    """Initialize ALPR engine. Can be called at startup or retried later."""
    global alpr, alpr_error, _ov_device

    try:
        from fast_alpr import ALPR

        try:
            import onnxruntime as ort
            available_eps = ort.get_available_providers()
            print(f"INFO: ONNX Runtime providers: {available_eps}")
            if "OpenVINOExecutionProvider" in available_eps:
                print(f"INFO: OpenVINO EP available - ORT_OPENVINO_DEVICE={_ov_device}")
                if _ov_device == "GPU":
                    print("INFO: Intel GPU acceleration enabled via OpenVINO EP")
            else:
                _ov_device = "CPU"
                print("INFO: OpenVINO EP not available - using CPU only")
        except Exception as e:
            print(f"INFO: ONNX Runtime check failed ({e}) - using {_ov_device}")

        alpr = ALPR(
            detector_model="yolo-v9-t-640-license-plate-end2end",
            ocr_model="cct-s-v1-global-model",
        )
        alpr_error = None
        print(f"INFO: ALPR v{VERSION} initialized - yolo-v9-t-640 + cct-s-v1 OCR (device: {_ov_device})")
        return True

    except ImportError as e:
        alpr = None
        alpr_error = f"ImportError: {e}"
        print(f"ERROR: fast_alpr could not be imported: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        alpr = None
        alpr_error = f"{type(e).__name__}: {e}"
        print(f"ERROR: fast_alpr failed to initialize: {e}")
        traceback.print_exc()
        return False

# Try to initialize at startup
print(f"INFO: LicensePlateControl Engine v{VERSION} starting...")
init_alpr()

app = FastAPI(title="LicensePlateControl Engine")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if alpr:
        try:
            start_time = time.time()
            results = alpr.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0

            output = []
            for res in results:
                plate_text = ""
                confidence = 0.0
                crop_b64 = None
                bbox = None

                if hasattr(res, 'ocr') and hasattr(res.ocr, 'text'):
                    plate_text = res.ocr.text
                    confidence = getattr(res.ocr, 'confidence', 0.0)
                elif isinstance(res, dict):
                    plate_text = res.get('plate', "")
                    confidence = res.get('confidence', 0.0)

                # Extract bounding box and crop plate region
                if hasattr(res, 'detection'):
                    det = res.detection
                    if hasattr(det, 'bounding_box'):
                        bb = det.bounding_box
                        # bounding_box may be [x1,y1,x2,y2] or an object
                        if hasattr(bb, 'x1'):
                            x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
                        elif isinstance(bb, (list, tuple)) and len(bb) == 4:
                            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                        else:
                            x1, y1, x2, y2 = None, None, None, None

                        if x1 is not None:
                            h, w = img.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            bbox = [x1, y1, x2, y2]
                            crop = img[y1:y2, x1:x2]
                            if crop.size > 0:
                                _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

                plate_text = str(plate_text).strip()
                if not plate_text:
                    plate_text = "UNKNOWN"

                entry = {
                    "plate": plate_text,
                    "confidence": confidence,
                }
                if bbox:
                    entry["bbox"] = bbox
                if crop_b64:
                    entry["crop_b64"] = crop_b64
                output.append(entry)
            return {"results": output, "processing_time_ms": processing_time_ms}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ALPR prediction failed: {str(e)}")
    else:
        return {"results": [{"plate": "MOCK-000", "confidence": 0.99}], "mock": True, "mock_reason": alpr_error, "processing_time_ms": 0.5}

@app.get("/health")
def health():
    return {
        "status": "ok" if alpr else "degraded",
        "alpr_loaded": alpr is not None,
        "error": alpr_error,
        "mock_mode": alpr is None,
        "device": _ov_device,
        "version": VERSION,
    }

@app.post("/reload")
def reload_alpr():
    """Retry ALPR initialization (e.g. after fixing GPU passthrough)."""
    success = init_alpr()
    return {
        "success": success,
        "alpr_loaded": alpr is not None,
        "error": alpr_error,
        "version": VERSION,
    }
