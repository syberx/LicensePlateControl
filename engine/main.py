from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import time

try:
    from fast_alpr import ALPR
    # Use the models from fast-alpr HuggingFace demo that work well for European plates:
    # Detector: yolo-v9-t-640 = 640px input, great accuracy for plate localization
    # OCR: cct-s-v1 = "small" model (better than default "xs"), trained on global plates
    import onnxruntime as ort
    
    # ---------------------------------------------------------
    # MONKEY-PATCH for OpenVINO ONNXRuntime
    # Forces open-image-models to use the Intel GPU via OpenVINO
    # instead of rendering on the CPU.
    # ---------------------------------------------------------
    original_InferenceSession = ort.InferenceSession

    def patched_InferenceSession(path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        if providers and "OpenVINOExecutionProvider" in providers:
            print(f"INFO: Intercepted ONNX Session Creation for {path_or_bytes}. Forcing Intel iGPU settings!")
            # OpenVINO specific options to target Intel GPU (iHD/i965)
            # If the user has an Intel CPU with integrated graphics, this offloads the heavy YOLOv9 network.
            forced_options = [{
                'device_type': 'GPU',
                'precision': 'FP16'
            }]
            # Pad the rest of the providers with empty options if there are multiple providers
            provider_opt_list = forced_options + [None] * (len(providers) - 1)
            return original_InferenceSession(path_or_bytes, sess_options, providers, provider_options=provider_opt_list, **kwargs)
        
        return original_InferenceSession(path_or_bytes, sess_options, providers, provider_options, **kwargs)

    ort.InferenceSession = patched_InferenceSession
    # ---------------------------------------------------------

    alpr = ALPR(
        detector_model="yolo-v9-t-640-license-plate-end2end",
        ocr_model="cct-s-v1-global-model",
    )
    print("INFO: ALPR initialized with yolo-v9-t-640 detector + cct-s-v1 OCR")
except ImportError as e:
    print(f"Warning: fast_alpr could not be imported: {e}. Using mock ALPR.")
    alpr = None
except Exception as e:
    print(f"Warning: fast_alpr failed to initialize: {e}. Using mock ALPR.")
    alpr = None

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

                # Extract from ALPRResult.ocr.text (primary path for fast-alpr)
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
        return {"results": [{"plate": "MOCK-000", "confidence": 0.99, "bounding_box": "10,10,100,50"}], "mock": True, "processing_time_ms": 15.0}

@app.get("/health")
def health():
    return {"status": "ok", "alpr_loaded": alpr is not None}
