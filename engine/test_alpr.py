import cv2
from fast_alpr import ALPR

alpr = ALPR(detector_model="yolo-v9-t-256-license-plate-end2end", ocr_model="cct-s-v1-global-model")
img = cv2.imread("../marcel_test.jpg")
res = alpr.predict(img)
for r in res:
    print(dir(r))
    print("detection:", getattr(r, 'detection', None))
    d = getattr(r, 'detection', None)
    if d:
        print("detection dir:", dir(d))
        print("bbox:", getattr(d, 'bounding_box', None), getattr(d, 'bbox', None))
        b = getattr(d, 'bounding_box', None)
        if b:
            print("bbox dir:", dir(b))
            print("x1:", getattr(b, 'x1', None))
    print("\n")
