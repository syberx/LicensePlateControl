# LicensePlateControl

Automatische Kennzeichenerkennung (ALPR) für die eigene Einfahrt — 100% lokal, Docker-basiert, DSGVO-konform.

Erkennt Nummernschilder über RTSP-Kamerastream oder Bilddateien (z.B. Blue Iris), gleicht sie gegen eine Whitelist ab und löst bei bekannten Kennzeichen Aktionen über Home Assistant oder MQTT aus (z.B. Garagentor öffnen).

![LicensePlateControl Dashboard](https://raw.githubusercontent.com/syberx/LicensePlateControl/main/docs/dashboard.png)

## Features

- **100% Lokal & Offline:** Keine Cloud. Die Erkennung läuft komplett lokal per YOLO v9 (Detection) + CCT OCR (Texterkennung).
- **RTSP Live-Stream:** Analysiert einen RTSP-Kamerastream in Echtzeit mit 2-Pass-Pipeline (YOLO Detection auf 320px → OCR auf Hi-Res Crop).
- **Docker-Ready:** 4 Container (Backend, Engine, Frontend, PostgreSQL) über Docker Compose.
- **Home Assistant & MQTT:** Löst bei erlaubtem Kennzeichen sofort einen Service-Call oder MQTT-Publish aus.
- **Motion Burst Processing:** Erkennt Bilderserien (z.B. von Blue Iris) und gruppiert sie zu einem Event ("Fast-First"-Verarbeitung).
- **Fuzzy Matching:** Fehlertolerante Erkennung (Levenshtein-Distanz) gleicht OCR-Fehler automatisch aus.
- **ROI-Masking:** Erkennungsbereich im Stream per Polygon einschränken — weniger Fehlerkennungen, schnellere Detection.
- **Modernes Dashboard:** Alpine.js Frontend im Dark-Glassmorphism-Design mit Pipeline-Debug-Tools.

---

## Systemarchitektur

| Container | Funktion |
|-----------|----------|
| **`postgres`** | PostgreSQL 16 — Events, Bilder (BYTEA), Konfiguration |
| **`engine`** | Fast ALPR Engine — YOLO v9-t-640 Detection + CCT-s-v1 OCR |
| **`backend`** | FastAPI — RTSP-Grabber, File-Watcher, Fuzzy-Matching, HA/MQTT |
| **`frontend`** | Nginx + Alpine.js Dashboard |

### 2-Pass RTSP Pipeline

```
RTSP Frame → ROI Crop+Mask → Resize 320px → YOLO /detect
                                                  ↓ (BBox gefunden)
                                   Hi-Res Crop aus Original → /ocr → Fuzzy Match → HA/MQTT
```

---

## Installation

### Voraussetzungen
- Docker und Docker Compose
- Optional: Ordner für eingehende Bilder (z.B. per FTP von Blue Iris)

### Starten

```bash
git clone https://github.com/syberx/LicensePlateControl.git
cd LicensePlateControl
docker compose up -d
```

Dashboard: `http://<server-ip>`

> **Wichtig:** `docker compose down -v` löscht die Datenbank! Für normale Neustarts `docker compose restart` verwenden.

---

## Kamera-Setup

### RTSP Stream (empfohlen)
Im Dashboard unter **Einstellungen → RTSP Stream** die URL eingeben (z.B. `rtsp://user:pass@192.168.1.100:554/stream`). Der Stream wird direkt analysiert — kein Datei-Export nötig.

### Blue Iris / Datei-Export
Das System überwacht den `/events/` Ordner. Dateinamen müssen einen Zeitstempel enthalten: `YYYYMMDD_HHMMSS` (z.B. `Cam_20260223_150409.jpg`).

Bilder mit < 9s Abstand werden zu einem Event gruppiert. Das erste Bild wird sofort ausgewertet ("Fast-First"), weitere Bilder verbessern das Ergebnis.

---

## Integrationen

### Home Assistant
Service-Call bei erlaubtem Kennzeichen. URL und Token in den Einstellungen konfigurieren.

### MQTT
Publiziert erkannte Kennzeichen als JSON auf ein konfigurierbares Topic.

---

## Datenschutz (DSGVO)

Für den Einsatz auf **privatem Grund** (Einfahrt, Garage) konzipiert. 100% lokal — keine Cloud-Uploads.
