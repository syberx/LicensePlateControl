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

VERSION = "1.5.1"

alpr = None
alpr_error = None
_detector = None
_ocr_model = None
_ov_device = os.environ.get("ORT_OPENVINO_DEVICE", "CPU")

# Stats counters
_stats = {
    "detect_calls": 0,
    "detect_hits": 0,          # calls that returned >= 1 detection
    "detect_errors": 0,
    "ocr_calls": 0,
    "ocr_hits": 0,             # calls that returned a plate text
    "ocr_errors": 0,
    "analyze_calls": 0,
    "analyze_hits": 0,
    "avg_detect_ms": 0.0,
    "avg_ocr_ms": 0.0,
    "avg_analyze_ms": 0.0,
    "two_pass_mode": "unknown", # "native" | "fallback" | "unavailable"
}
_detect_times = []
_ocr_times = []
_analyze_times = []


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

        # Introspect ALPR object to find detector and OCR components
        print(f"INFO: ALPR object type: {type(alpr).__name__}")
        print(f"INFO: ALPR attributes: {[a for a in dir(alpr) if not a.startswith('__')]}")

        # Try multiple known attribute names for detector
        for attr_name in ['detector', '_detector', 'plate_detector', 'det_model', 'det']:
            candidate = getattr(alpr, attr_name, None)
            if candidate is not None:
                _detector = candidate
                print(f"INFO: Found detector at alpr.{attr_name} — type: {type(_detector).__name__}")
                if hasattr(_detector, 'predict'):
                    print(f"INFO: Detector has .predict() method")
                break

        # Try multiple known attribute names for OCR
        for attr_name in ['ocr', '_ocr', 'ocr_model', 'plate_ocr', 'recognizer']:
            candidate = getattr(alpr, attr_name, None)
            if candidate is not None:
                _ocr_model = candidate
                print(f"INFO: Found OCR at alpr.{attr_name} — type: {type(_ocr_model).__name__}")
                if hasattr(_ocr_model, 'predict'):
                    print(f"INFO: OCR model has .predict() method")
                break

        alpr_error = None
        two_pass = _detector is not None and _ocr_model is not None
        if two_pass:
            _stats["two_pass_mode"] = "native"
        else:
            _stats["two_pass_mode"] = "fallback"

        print(f"INFO: ALPR v{VERSION} initialized - yolo-v9-t-640 + cct-s-v1 OCR (device: {_ov_device})")
        print(f"INFO: 2-pass mode: {'NATIVE' if two_pass else 'FALLBACK (using alpr.predict for both endpoints)'}")
        return True

    except ImportError as e:
        alpr = None
        alpr_error = f"ImportError: {e}"
        _stats["two_pass_mode"] = "unavailable"
        print(f"ERROR: fast_alpr could not be imported: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        alpr = None
        alpr_error = f"{type(e).__name__}: {e}"
        _stats["two_pass_mode"] = "unavailable"
        print(f"ERROR: fast_alpr failed to initialize: {e}")
        traceback.print_exc()
        return False


def _extract_bbox(det):
    """Extract bounding box [x1,y1,x2,y2] from a detection result object."""
    # Try .bounding_box attribute (fast_alpr standard)
    if hasattr(det, 'bounding_box'):
        bb = det.bounding_box
        if hasattr(bb, 'x1'):
            return [int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)]
        elif isinstance(bb, (list, tuple)) and len(bb) == 4:
            return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
    # Try .bbox attribute
    if hasattr(det, 'bbox'):
        bb = det.bbox
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
    # Try direct list/tuple
    if isinstance(det, (list, tuple)) and len(det) >= 4:
        return [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
    return None


def _extract_det_confidence(det):
    """Extract confidence from a detection result object."""
    for attr in ['confidence', 'conf', 'score']:
        val = getattr(det, attr, None)
        if val is not None:
            return float(val)
    if isinstance(det, dict):
        for key in ['confidence', 'conf', 'score']:
            if key in det:
                return float(det[key])
    return 0.0


def run_selftest():
    """Run a startup self-test with a synthetic license plate image."""
    if not alpr:
        print("SELFTEST: SKIPPED — ALPR not loaded (mock mode)")
        return

    print("SELFTEST: Running startup self-test with synthetic plate image...")
    try:
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (80, 80, 80)
        plate_x1, plate_y1 = 150, 180
        plate_x2, plate_y2 = 490, 300
        cv2.rectangle(test_img, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 255, 255), -1)
        cv2.rectangle(test_img, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_img, "MK AB 123", (170, 265), font, 1.8, (0, 0, 0), 4, cv2.LINE_AA)

        # Benchmark: full pipeline
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

                # Introspect result structure for debugging
                print(f"SELFTEST: Result type: {type(res).__name__}, attrs: {[a for a in dir(res) if not a.startswith('_')]}")
                if hasattr(res, 'detection'):
                    det = res.detection
                    print(f"SELFTEST: Detection type: {type(det).__name__}, attrs: {[a for a in dir(det) if not a.startswith('_')]}")
                    bbox = _extract_bbox(det)
                    print(f"SELFTEST: BBox: {bbox}")
        else:
            print(f"SELFTEST: WARN — No plate detected ({elapsed_ms:.0f}ms)")

        # Benchmark: detection only (if available)
        if _detector:
            try:
                start_det = time.time()
                det_results = _detector.predict(test_img)
                det_ms = (time.time() - start_det) * 1000.0
                det_count = len(det_results) if det_results else 0
                print(f"SELFTEST: Detect-only: {det_count} plates in {det_ms:.0f}ms")
                if det_results:
                    det0 = det_results[0]
                    print(f"SELFTEST: Det[0] type: {type(det0).__name__}, attrs: {[a for a in dir(det0) if not a.startswith('_')]}")
                    bbox = _extract_bbox(det0)
                    conf = _extract_det_confidence(det0)
                    print(f"SELFTEST: Det[0] bbox={bbox}, conf={conf}")

                # Benchmark: OCR only
                if _ocr_model and det_results:
                    bbox = _extract_bbox(det_results[0])
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        crop = test_img[max(0,y1):y2, max(0,x1):x2]
                        if crop.size > 0:
                            start_ocr = time.time()
                            ocr_result = _ocr_model.predict(crop)
                            ocr_ms = (time.time() - start_ocr) * 1000.0
                            print(f"SELFTEST: OCR result type: {type(ocr_result).__name__}")
                            ocr_text = ""
                            if ocr_result and hasattr(ocr_result, 'text'):
                                ocr_text = ocr_result.text
                            elif isinstance(ocr_result, str):
                                ocr_text = ocr_result
                            print(f"SELFTEST: OCR-only on crop: '{ocr_text}' in {ocr_ms:.0f}ms")
                            print(f"SELFTEST: 2-pass total: {det_ms+ocr_ms:.0f}ms (detect={det_ms:.0f} + ocr={ocr_ms:.0f}) vs single-pass={elapsed_ms:.0f}ms")
            except Exception as e:
                print(f"SELFTEST: Detect-only FAILED: {e}")
                traceback.print_exc()

        print(f"SELFTEST: Pipeline operational — {elapsed_ms:.0f}ms total")

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


def _update_avg(times_list, new_ms, max_entries=100):
    """Update a rolling average."""
    times_list.append(new_ms)
    if len(times_list) > max_entries:
        times_list.pop(0)
    return sum(times_list) / len(times_list)


@app.post("/detect")
async def detect_only(file: UploadFile = File(...)):
    """Pass 1: YOLO detection only — returns bounding boxes, no OCR.
    Designed for 640px downscaled images. Fast (~100-200ms).
    Falls back to alpr.predict() if native detector not available."""
    _stats["detect_calls"] += 1
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        _stats["detect_errors"] += 1
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Try native detector first (fast, detection only)
    if _detector and hasattr(_detector, 'predict'):
        try:
            start_time = time.time()
            det_results = _detector.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0
            _stats["avg_detect_ms"] = _update_avg(_detect_times, processing_time_ms)

            detections = []
            if det_results:
                for det in det_results:
                    bbox = _extract_bbox(det)
                    det_conf = _extract_det_confidence(det)
                    if bbox:
                        detections.append({"bbox": bbox, "confidence": det_conf})

            if detections:
                _stats["detect_hits"] += 1
            return {"detections": detections, "processing_time_ms": processing_time_ms, "mode": "native"}
        except Exception as e:
            _stats["detect_errors"] += 1
            print(f"ERROR: Native detector failed: {e}, falling back to alpr.predict()")
            traceback.print_exc()

    # Fallback: use full ALPR pipeline but only return bboxes
    if alpr:
        try:
            start_time = time.time()
            results = alpr.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0
            _stats["avg_detect_ms"] = _update_avg(_detect_times, processing_time_ms)

            detections = []
            for res in results:
                if hasattr(res, 'detection'):
                    bbox = _extract_bbox(res.detection)
                    det_conf = _extract_det_confidence(res.detection)
                    if bbox:
                        detections.append({"bbox": bbox, "confidence": det_conf})

            if detections:
                _stats["detect_hits"] += 1
            return {"detections": detections, "processing_time_ms": processing_time_ms, "mode": "fallback"}
        except Exception as e:
            _stats["detect_errors"] += 1
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    else:
        return {"detections": [{"bbox": [100, 100, 300, 200], "confidence": 0.99}], "mock": True, "processing_time_ms": 0.5, "mode": "mock"}


@app.post("/ocr")
async def ocr_only(file: UploadFile = File(...)):
    """Pass 2: OCR only on a plate crop image. Fast (~50-100ms).
    Falls back to alpr.predict() if native OCR not available."""
    _stats["ocr_calls"] += 1
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        _stats["ocr_errors"] += 1
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Try native OCR first (fast, OCR only)
    if _ocr_model and hasattr(_ocr_model, 'predict'):
        try:
            start_time = time.time()
            ocr_result = _ocr_model.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0
            _stats["avg_ocr_ms"] = _update_avg(_ocr_times, processing_time_ms)

            plate_text = ""
            confidence = 0.0
            if ocr_result:
                if hasattr(ocr_result, 'text'):
                    plate_text = str(ocr_result.text).strip()
                    confidence = float(getattr(ocr_result, 'confidence', 0.0))
                elif isinstance(ocr_result, str):
                    plate_text = ocr_result.strip()
                elif isinstance(ocr_result, dict):
                    plate_text = str(ocr_result.get('text', '')).strip()
                    confidence = float(ocr_result.get('confidence', 0.0))

            if not plate_text:
                plate_text = "UNKNOWN"
            else:
                _stats["ocr_hits"] += 1

            return {"plate": plate_text, "confidence": confidence, "processing_time_ms": processing_time_ms, "mode": "native"}
        except Exception as e:
            _stats["ocr_errors"] += 1
            print(f"ERROR: Native OCR failed: {e}, falling back to alpr.predict()")
            traceback.print_exc()

    # Fallback: use full ALPR pipeline on the crop
    if alpr:
        try:
            start_time = time.time()
            results = alpr.predict(img)
            processing_time_ms = (time.time() - start_time) * 1000.0
            _stats["avg_ocr_ms"] = _update_avg(_ocr_times, processing_time_ms)

            plate_text = "UNKNOWN"
            confidence = 0.0
            for res in results:
                if hasattr(res, 'ocr') and hasattr(res.ocr, 'text'):
                    plate_text = str(res.ocr.text).strip()
                    confidence = float(getattr(res.ocr, 'confidence', 0.0))
                    break
            if plate_text and plate_text != "UNKNOWN":
                _stats["ocr_hits"] += 1
            return {"plate": plate_text, "confidence": confidence, "processing_time_ms": processing_time_ms, "mode": "fallback"}
        except Exception as e:
            _stats["ocr_errors"] += 1
            raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
    else:
        return {"plate": "MOCK-OCR", "confidence": 0.99, "mock": True, "processing_time_ms": 0.5, "mode": "mock"}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Single-pass: full detection + OCR in one call. Used by file watcher and as fallback."""
    _stats["analyze_calls"] += 1
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
            _stats["avg_analyze_ms"] = _update_avg(_analyze_times, processing_time_ms)

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

                entry = {"plate": plate_text, "confidence": confidence}
                if bbox:
                    entry["bbox"] = bbox
                if crop_b64:
                    entry["crop_b64"] = crop_b64
                output.append(entry)

            if output:
                _stats["analyze_hits"] += 1
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
        "two_pass_mode": _stats["two_pass_mode"],
        "error": alpr_error,
        "mock_mode": alpr is None,
        "device": _ov_device,
        "version": VERSION,
        "stats": _stats,
    }


@app.get("/stats")
def get_stats():
    """Detailed performance statistics."""
    return {
        "version": VERSION,
        "two_pass_mode": _stats["two_pass_mode"],
        "detector_available": _detector is not None,
        "ocr_available": _ocr_model is not None,
        "detect": {
            "calls": _stats["detect_calls"],
            "hits": _stats["detect_hits"],
            "errors": _stats["detect_errors"],
            "hit_rate": round(_stats["detect_hits"] / max(_stats["detect_calls"], 1) * 100, 1),
            "avg_ms": round(_stats["avg_detect_ms"], 1),
        },
        "ocr": {
            "calls": _stats["ocr_calls"],
            "hits": _stats["ocr_hits"],
            "errors": _stats["ocr_errors"],
            "hit_rate": round(_stats["ocr_hits"] / max(_stats["ocr_calls"], 1) * 100, 1),
            "avg_ms": round(_stats["avg_ocr_ms"], 1),
        },
        "analyze": {
            "calls": _stats["analyze_calls"],
            "hits": _stats["analyze_hits"],
            "avg_ms": round(_stats["avg_analyze_ms"], 1),
        },
    }


@app.post("/reload")
def reload_alpr():
    """Retry ALPR initialization."""
    success = init_alpr()
    return {
        "success": success,
        "alpr_loaded": alpr is not None,
        "two_pass_available": _detector is not None and _ocr_model is not None,
        "two_pass_mode": _stats["two_pass_mode"],
        "error": alpr_error,
        "version": VERSION,
    }
