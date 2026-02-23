# LicensePlateControl

Ein lokales, datenschutzfreundliches Open-Source-System zur automatischen Kennzeichenerkennung (ALPR) für die eigene Einfahrt oder Garage. 

Das System ist dafür gedacht, Bilder von Überwachungskameras (z.B. via Blue Iris) entgegenzunehmen, Nummernschilder DSGVO-konform und lokal (ohne Cloud) auszulesen und bei bekannten Kennzeichen Aktionen über [Home Assistant](https://www.home-assistant.io/) auszulösen (z.B. Garagentor öffnen).

![LicensePlateControl Dashboard](https://raw.githubusercontent.com/syberx/LicensePlateControl/main/docs/dashboard.png)

## Features

- **100% Lokal & Offline:** Keine Cloud-Anbindung nötig. Die Erkennung läuft komplett lokal per neuronalem Netz (CNN).
- **Docker-Ready:** Einfaches Setup über Docker Compose mit separaten Containern für Backend, Frontend, Datenbank und ALPR-Engine.
- **Home Assistant Integration:** Löst bei erkanntem "erlaubten" Kennzeichen sofort einen Webhook oder Service-Call in Home Assistant aus.
- **Motion Burst Processing:** Erkennt automatische Bilderserien (z.B. 10 Bilder pro Sekunde von Blue Iris) und gruppiert sie intelligent zu einem Event zusammen ("Fast-First"-Verarbeitung).
- **Fuzzy Matching:** Fehlertolerante Kennzeichenerkennung (Levenshtein-Distanz) gleicht leichte OCR-Fehler (z.B. O statt 0) automatisch aus.
- **Datenbank & Speicherung:** Speichert erkannte Events und die dazugehörigen Bilder effizient in einer PostgreSQL-Datenbank (Bilder werden nach der Verarbeitung automatisch aus dem Hotfolder gelöscht).
- **Modernes Dashboard:** Responsive Alpine.js Frontend im schicken "Dark Glassmorphism"-Design zur Verwaltung von Kennzeichen, Events und Einstellungen.

---

## Systemarchitektur

Das System besteht aus 4 Docker-Containern:

1. **`postgres`**: Die relationale Datenbank (PostgreSQL 16) speichert Events, Bilder (als BYTEA) und Konfigurationen.
2. **`engine`**: Ein Python-Microservice, der die eigentliche Bilderkennung (ALPR) durchführt.
3. **`backend`**: FastAPI Applikation, die den Watch-Folder überwacht (`/events/`), mit der DB kommuniziert, die Logik für "Motion Bursts" enthält und Home Assistant ansteuert.
4. **`frontend`**: Nginx Webserver, der das Alpine.js Dashboard bereitstellt.

---

## 🚀 Installation & Setup

### Voraussetzungen
- Docker und Docker Compose installiert
- Ein Ordner für eingehende Bilder (z.B. per FTP von der Kamera oder von Blue Iris)

### 1. Repository klonen
```bash
git clone https://github.com/syberx/LicensePlateControl.git
cd LicensePlateControl
```

### 2. Konfiguration (Optional)
Im Frontend unter "Einstellungen" (oder direkt über die API) können später die Home Assistant URL (z.B. `http://homeassistant.local:8123`) und das Long-Lived Access Token hinterlegt werden.

### 3. Container starten
Das System wird komplett über Docker Compose hochgefahren:

```bash
docker compose up -d
```

> ⚠️ **Achtung zu Updates und Datenbanken:** Wenn du das System per `docker compose down -v` stoppst, wird das Datenbank-Volume (`licenseplatecontrol_pgdata`) gelöscht und alle gespeicherten Kennzeichen und Events gehen verloren! Nutze für normale Neustarts nur `docker compose restart` oder `docker compose down` (ohne `-v`).

Das Dashboard ist danach unter `http://localhost` (oder der IP deines Servers) erreichbar.

---

## 📷 Kamera / Blue Iris Setup (WICHTIG!)

Das System reagiert automatisch auf neue Bilder in dem als Docker-Volume gemounteten `/events/` Ordner. 

### Zeitstempel-Format (Dateinamen)
Damit das System zusammenhängende "Motion Bursts" (eine Serie von Bildern, während ein Auto durchs Bild fährt) korrekt als **ein Event** gruppieren kann, **muss** der Dateiname einen Zeitstempel enthalten.

Das System erwartet folgendes Format im Dateinamen:
`YYYYMMDD_HHMMSS` (z.B. `Aufnahme_20260223_150409.jpg`)

**In Blue Iris:**
Setze das Format für den Datei-Export bei Trigger z.B. auf:
`&CAM.%Y%m%d_%H%M%S`

**Wie die Gruppierung funktioniert:**
- Das Backend liest die Zeitstempel aus den Dateinamen.
- Bilder, die einen zeitlichen Abstand von **weniger als 9 Sekunden** haben, werden zu einem Event (Motion Burst) zusammengefasst.
- Das **erste** Bild einer Serie wird per "Fast-First" sofort ausgewertet, um Home Assistant verzögerungsfrei zu triggern.
- Alle weiteren Bilder der Serie werden als "Follow-ups" verarbeitet und dem Event in der Datenbank hinzugefügt. Das beste Kennzeichen gewinnt.

---

## Home Assistant Integration

Wenn ein erlaubtes Kennzeichen sicher erkannt wurde, triggert das Backend Home Assistant. Du musst in Home Assistant einen entsprechenden Automatisierungs-Trigger bereithalten.

Beispiel für den Standard-Callout in den LicensePlateControl Settings:
- **HA Service:** `switch.turn_on`
- **Target (im Script):** Wird vom Backend per JSON-Payload an HA geschickt. 

Das Backend ruft die Home Assistant REST API auf: `POST /api/services/<domain>/<service>`

---

## Datenschutz (DSGVO) Hinweis

Dieses Tool ist für den Einsatz auf **privatem Grund** (Einfahrt, Garage, Carport) konzipiert. Bitte achte darauf, dass deine Kameras keinen öffentlichen Raum filmen. Da das System 100% lokal läuft, werden keine Bilder in eine Cloud hochgeladen, was die datenschutzrechtliche Situation für den Privatgebrauch erheblich vereinfacht.
