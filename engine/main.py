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

VERSION = "1.5.0"

alpr = None
alpr_error = None
_detector = None
_ocr_model = None
_ov_device = os.environ.get("ORT_OPENVINO_DEVICE", "CPU")

def init_alpr():
    """Initialize ALPR engine. Can be called at startup or retried later."""
    global alpr, alpr_error, _ov_device, _detector, _ocr_model

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

        # Extract individual components for 2-pass architecture
        _detector = getattr(alpr, 'detector', None)
        _ocr_model = getattr(alpr, 'ocr', None)

        alpr_error = None
        two_pass = _detector is not None and _ocr_model is not None
        print(f"INFO: ALPR v{VERSION} initialized - yolo-v9-t-640 + cct-s-v1 OCR (device: {_ov_device})")
        print(f"INFO: 2-pass mode: {'AVAILABLE' if two_pass else 'UNAVAILABLE (falling back to single-pass)'}")
        if _detector:
            print(f"INFO: Detector type: {type(_detector).__name__}")
        if _ocr_model:
            print(f"INFO: OCR model type: {type(_ocr_model).__name__}")
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


def _extract_bbox(det):
    """Extract bounding box [x1,y1,x2,y2] from a detection result."""
    if hasattr(det, 'bounding_box'):
        bb = det.bounding_box
        if hasattr(bb, 'x1'):
            return [int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)]
        elif isinstance(bb, (list, tuple)) and len(bb) == 4:
            return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
    return None


def run_selftest():
    """Run a startup self-test with a synthetic license plate image.
    Benchmarks detection-only and OCR-only separately to confirm 2-pass pipeline."""
    if not alpr:
        print("SELFTEST: SKIPPED — ALPR not loaded (mock mode)")
        return

    print("SELFTEST: Running startup self-test with synthetic plate image...")
    try:
        # Create a 640x480 image with a white plate region
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (80, 80, 80)  # Dark gray background

        # Draw a white plate rectangle
        plate_x1, plate_y1 = 150, 180
        plate_x2, plate_y2 = 490, 300
        cv2.rectangle(test_img, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 255, 255), -1)
        cv2.rectangle(test_img, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 0), 3)

        # Add plate text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_img, "MK AB 123", (170, 265), font, 1.8, (0, 0, 0), 4, cv2.LINE_AA)

        # Benchmark: full pipeline (detect+OCR)
        start = time.time()
        results = alpr.predict(test_img)
        elapsed_ms = (time.time() - start) * 1000.0

        if results:
            for res in results:
                plate_text = ""
                conf = 0.0
                if hasattr(res, 'ocr') and hasattr(res.ocr, 'text'):
                    plate_text = res.ocr.text
                    conf = getattr(res.ocr, 'confidence', 0.0)
                print(f"SELFTEST: OK — Detected '{plate_text}' (conf={conf:.2f}) in {elapsed_ms:.0f}ms")
        else:
            print(f"SELFTEST: WARN — No plate detected on synthetic image ({elapsed_ms:.0f}ms). "
                  "This is normal — synthetic images may not match trained plate patterns.")

        # Benchmark: detection only (if 2-pass available)
        if _detector:
            start_det = time.time()
            det_results = _detector.predict(test_img)
            det_ms = (time.time() - start_det) * 1000.0
            print(f"SELFTEST: Detect-only: {len(det_results) if det_results else 0} plates in {det_ms:.0f}ms")

            # Benchmark: OCR only on a crop (if we got a detection)
            if _ocr_model and det_results:
                for det in det_results:
                    bbox = _extract_bbox(det)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        crop = test_img[max(0,y1):y2, max(0,x1):x2]
                        if crop.size > 0:
                            start_ocr = time.time()
                            ocr_result = _ocr_model.predict(crop)
                            ocr_ms = (time.time() - start_ocr) * 1000.0
                            ocr_text = ""
                            if ocr_result and hasattr(ocr_result, 'text'):
                                ocr_text = ocr_result.text
                            print(f"SELFTEST: OCR-only on crop: '{ocr_text}' in {ocr_ms:.0f}ms")
                            print(f"SELFTEST: 2-pass total: {det_ms:.0f}ms detect + {ocr_ms:.0f}ms OCR = {det_ms+ocr_ms:.0f}ms (vs {elapsed_ms:.0f}ms single-pass)")
                            break

        print(f"SELFTEST: Pipeline operational (YOLO detector + CCT OCR) — {elapsed_ms:.0f}ms total")

    except Exception as e:
        print(f"SELFTEST: ERROR — {e}")
        traceback.print_exc()


# Try to initialize at startup
print(f"INFO: LicensePlateControl Engine v{VERSION} starting...")
print(f"INFO: Detector: yolo-v9-t-640-license-plate-end2end")
print(f"INFO: OCR Model: cct-s-v1-global-model")
print(f"INFO: Device: {_ov_device}")
init_alpr()
run_selftest()

app = FastAPI(title="LicensePlateControl Engine")


@app.post("/detect")
async def detect_only(file: UploadFile = File(...)):
    """Pass 1: YOLO detection only — returns bounding boxes, no OCR.
    Designed for 640px downscaled images. Fast (~100-200ms)."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if _detector:
        try:
            start_time = time.time()
            det_results = _detector.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0

            detections = []
            for det in det_results:
                bbox = _extract_bbox(det)
                det_conf = float(getattr(det, 'confidence', 0.0)) if hasattr(det, 'confidence') else 0.0
                if bbox:
                    detections.append({"bbox": bbox, "confidence": det_conf})

            return {"detections": detections, "processing_time_ms": processing_time_ms}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    elif alpr:
        # Fallback: use full ALPR pipeline but only return bboxes
        try:
            start_time = time.time()
            results = alpr.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0
            detections = []
            for res in results:
                if hasattr(res, 'detection'):
                    bbox = _extract_bbox(res.detection)
                    det_conf = float(getattr(res.detection, 'confidence', 0.0)) if hasattr(res.detection, 'confidence') else 0.0
                    if bbox:
                        detections.append({"bbox": bbox, "confidence": det_conf})
            return {"detections": detections, "processing_time_ms": processing_time_ms, "fallback": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    else:
        return {"detections": [{"bbox": [100, 100, 300, 200], "confidence": 0.99}], "mock": True, "processing_time_ms": 0.5}


@app.post("/ocr")
async def ocr_only(file: UploadFile = File(...)):
    """Pass 2: OCR only on a plate crop image. Fast (~50-100ms)."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if _ocr_model:
        try:
            start_time = time.time()
            ocr_result = _ocr_model.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0

            plate_text = ""
            confidence = 0.0
            if ocr_result:
                if hasattr(ocr_result, 'text'):
                    plate_text = str(ocr_result.text).strip()
                    confidence = float(getattr(ocr_result, 'confidence', 0.0))
                elif isinstance(ocr_result, str):
                    plate_text = ocr_result.strip()

            if not plate_text:
                plate_text = "UNKNOWN"

            return {"plate": plate_text, "confidence": confidence, "processing_time_ms": processing_time_ms}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
    elif alpr:
        # Fallback: use full ALPR pipeline on the crop
        try:
            start_time = time.time()
            results = alpr.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0
            plate_text = "UNKNOWN"
            confidence = 0.0
            for res in results:
                if hasattr(res, 'ocr') and hasattr(res.ocr, 'text'):
                    plate_text = str(res.ocr.text).strip()
                    confidence = float(getattr(res.ocr, 'confidence', 0.0))
                    break
            return {"plate": plate_text, "confidence": confidence, "processing_time_ms": processing_time_ms, "fallback": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
    else:
        return {"plate": "MOCK-OCR", "confidence": 0.99, "mock": True, "processing_time_ms": 0.5}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Legacy single-pass: full detection + OCR in one call. Used by file watcher."""
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
                    bbox = _extract_bbox(det)

                    if bbox:
                        x1, y1, x2, y2 = bbox
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
        "two_pass_available": _detector is not None and _ocr_model is not None,
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
        "two_pass_available": _detector is not None and _ocr_model is not None,
        "error": alpr_error,
        "version": VERSION,
    }
