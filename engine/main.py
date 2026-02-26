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

                if hasattr(res, 'ocr') and hasattr(res.ocr, 'text'):
                    plate_text = res.ocr.text
                    confidence = getattr(res.ocr, 'confidence', 0.0)
                elif isinstance(res, dict):
                    plate_text = res.get('plate', "")
                    confidence = res.get('confidence', 0.0)

                plate_text = str(plate_text).strip()
                if not plate_text:
                    plate_text = "UNKNOWN"

                output.append({
                    "plate": plate_text,
                    "confidence": confidence
                })
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
