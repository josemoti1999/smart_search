#!/usr/bin/env bash
# build.sh — Full build: Python binary → Electron app → .dmg
# Run from the smart_search/ project root.
# Requirements: Python venv activated, Node + npm installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "════════════════════════════════════════"
echo "  Smart Search — macOS Build"
echo "════════════════════════════════════════"
echo ""


# ── Step 1: Activate venv ───────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtualenv activated"
else
    echo "✗ venv not found. Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi


# ── Step 2: Install PyInstaller if needed ───────
if ! python -m PyInstaller --version &>/dev/null; then
    echo "→ Installing PyInstaller…"
    pip install pyinstaller
fi
echo "✓ PyInstaller ready"


# ── Step 3: Build Python binary ─────────────────
echo ""
echo "→ Building Python binary with PyInstaller…"
python -m PyInstaller smartsearch.spec --clean --noconfirm

if [ ! -f "dist/smartsearch" ]; then
    echo "✗ PyInstaller build failed — dist/smartsearch not found"
    exit 1
fi

chmod +x dist/smartsearch
echo "✓ Python binary: dist/smartsearch ($(du -sh dist/smartsearch | cut -f1))"


# ── Step 4: Generate tray icon if not present ───
ICON_SRC="electron/assets/tray-icon.png"
if [ ! -f "$ICON_SRC" ]; then
    echo "→ Generating placeholder tray icon…"
    python3 - <<'PYEOF'
import struct, zlib, base64

# Minimal 16x16 PNG — blue magnifying-glass circle
def make_png(w, h, rgba_rows):
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    raw = b''
    for row in rgba_rows:
        raw += b'\x00' + bytes([c for px in row for c in px])
    idat = chunk(b'IDAT', zlib.compress(raw))
    iend = chunk(b'IEND', b'')
    return sig + ihdr + idat + iend

import math
size = 16
rows = []
cx = cy = size / 2
for y in range(size):
    row = []
    for x in range(size):
        dx, dy = x - cx + 0.5, y - cy + 0.5
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 6.5 and dist > 4.0:
            row.append((59, 130, 246))  # blue ring
        elif dist < 4.0 and dist > 2.5:
            row.append((219, 234, 254))  # light fill
        elif 5.5 < dist < 7.5 and dx > 2 and dy > 2:
            row.append((59, 130, 246))  # handle
        else:
            row.append((0, 0, 0))
    rows.append(row)

import os
os.makedirs('electron/assets', exist_ok=True)
png = make_png(size, size, rows)
with open('electron/assets/tray-icon.png', 'wb') as f:
    f.write(png)
print("Generated tray-icon.png")
PYEOF
fi
echo "✓ Tray icon ready"


# ── Step 5: Install Electron dependencies ───────
echo ""
echo "→ Installing Electron dependencies…"
cd electron
npm install --silent
cd ..
echo "✓ Electron node_modules ready"


# ── Step 6: Build .dmg with electron-builder ────
echo ""
echo "→ Building macOS .dmg with electron-builder…"
cd electron
npm run dist
cd ..

DMG=$(find electron/dist -name "*.dmg" | head -1)
if [ -n "$DMG" ]; then
    echo ""
    echo "════════════════════════════════════════"
    echo "  BUILD COMPLETE"
    echo "  Installer: $DMG"
    echo "════════════════════════════════════════"
    echo ""
    open "$(dirname "$DMG")"
else
    echo ""
    echo "✗ electron-builder succeeded but no .dmg found."
    echo "  Check electron/dist/ for output."
fi
