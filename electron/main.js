'use strict';

const {
  app,
  BrowserWindow,
  Tray,
  Menu,
  ipcMain,
  dialog,
  nativeImage,
  shell,
} = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

const DEFAULT_PORT = 5001;

function getPort() {
  // Read actual runtime port written by app.py (handles port conflicts)
  try {
    const configPath = require('os').homedir() + '/.smartsearch/config.json';
    if (fs.existsSync(configPath)) {
      const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      return cfg._runtime_port || cfg.port || DEFAULT_PORT;
    }
  } catch(e) {}
  return DEFAULT_PORT;
}

const PORT = DEFAULT_PORT; // initial value; re-read after Python starts
const IS_DEV = process.argv.includes('--dev') || !app.isPackaged;

let tray = null;
let mainWindow = null;
let pythonProcess = null;


// ─────────────────────────────────────────────────
// Python sidecar helpers
// ─────────────────────────────────────────────────

function getPythonBinary() {
  if (app.isPackaged) {
    // PyInstaller one-file binary bundled as extra resource
    return path.join(process.resourcesPath, 'smartsearch');
  }
  // Development: use the virtualenv python + app.py
  return path.join(__dirname, '..', 'venv', 'bin', 'python3');
}

function startPythonServer() {
  const bin = getPythonBinary();

  let args = [];
  let env = { ...process.env };

  if (!app.isPackaged) {
    // Dev mode: run app.py with the venv interpreter
    args = [path.join(__dirname, '..', 'app.py')];
    // Load .env if present
    const envFile = path.join(__dirname, '..', '.env');
    if (fs.existsSync(envFile)) {
      const lines = fs.readFileSync(envFile, 'utf8').split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith('#')) {
          const eq = trimmed.indexOf('=');
          if (eq !== -1) {
            const k = trimmed.slice(0, eq).trim();
            const v = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, '');
            env[k] = v;
          }
        }
      }
    }
  }

  console.log(`[Electron] Starting Python: ${bin} ${args.join(' ')}`);

  pythonProcess = spawn(bin, args, {
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  pythonProcess.stdout.on('data', (d) => process.stdout.write(`[Python] ${d}`));
  pythonProcess.stderr.on('data', (d) => process.stderr.write(`[Python] ${d}`));
  pythonProcess.on('close', (code) => {
    console.log(`[Python] Process exited with code ${code}`);
  });
}

function isServerRunning(port) {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/progress`, (res) => resolve(true));
    req.setTimeout(800);
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

function waitForServer(port, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    function check() {
      const req = http.get(`http://127.0.0.1:${port}/progress`, (res) => {
        resolve();
      });
      req.setTimeout(800);
      req.on('error', () => {
        if (Date.now() > deadline) {
          reject(new Error('Python backend did not start within 30 seconds.'));
        } else {
          setTimeout(check, 600);
        }
      });
      req.on('timeout', () => { req.destroy(); });
    }
    check();
  });
}


// ─────────────────────────────────────────────────
// Window
// ─────────────────────────────────────────────────

function createWindow(port = DEFAULT_PORT) {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 800,
    minHeight: 600,
    title: 'Smart Search',
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#f9fafb',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false,
    },
    show: false,
  });

  mainWindow.loadURL(`http://127.0.0.1:${port}`);

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Hide instead of close (keep running in tray)
  mainWindow.on('close', (e) => {
    if (!app.isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });

  if (IS_DEV) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
}


// ─────────────────────────────────────────────────
// Tray
// ─────────────────────────────────────────────────

function getTrayIcon() {
  const iconPath = path.join(__dirname, 'assets', 'tray-icon.png');
  if (fs.existsSync(iconPath)) {
    return nativeImage.createFromPath(iconPath).resize({ width: 16, height: 16 });
  }
  // Fallback: tiny 1x1 transparent PNG
  return nativeImage.createEmpty();
}

function createTray(port = DEFAULT_PORT) {
  const icon = getTrayIcon();
  tray = new Tray(icon);
  tray.setToolTip('Smart Search');

  const menu = Menu.buildFromTemplate([
    {
      label: 'Open Smart Search',
      click: () => { mainWindow.show(); mainWindow.focus(); },
    },
    {
      label: 'Settings',
      click: () => {
        mainWindow.show();
        mainWindow.loadURL(`http://127.0.0.1:${port}/settings`);
        mainWindow.focus();
      },
    },
    { type: 'separator' },
    {
      label: 'Quit Smart Search',
      click: () => {
        app.isQuitting = true;
        app.quit();
      },
    },
  ]);

  tray.setContextMenu(menu);
  tray.on('click', () => { mainWindow.show(); mainWindow.focus(); });
}


// ─────────────────────────────────────────────────
// IPC handlers
// ─────────────────────────────────────────────────

ipcMain.handle('open-folder-dialog', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    title: 'Choose a folder to scan',
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('open-url', async (_, url) => {
  await shell.openExternal(url);
});


// ─────────────────────────────────────────────────
// App lifecycle
// ─────────────────────────────────────────────────

app.whenReady().then(async () => {
  // macOS: don't show in Dock while it's running as a tray app
  if (process.platform === 'darwin') {
    app.dock.hide();
  }

  // In dev mode, only spawn Python if it's not already running
  const alreadyUp = await isServerRunning(PORT);
  if (!alreadyUp) {
    startPythonServer();
  } else {
    console.log('[Electron] Python backend already running, skipping spawn.');
  }

  // Show a loading window while Python warms up
  const loadingWin = new BrowserWindow({
    width: 380,
    height: 220,
    resizable: false,
    frame: false,
    backgroundColor: '#f9fafb',
    webPreferences: { nodeIntegration: false },
  });
  loadingWin.loadURL('data:text/html,' + encodeURIComponent(`
    <html><head><style>
      body{font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#f9fafb;flex-direction:column;gap:16px;}
      .spinner{width:32px;height:32px;border:3px solid #e5e7eb;border-top:3px solid #3b82f6;border-radius:50%;animation:spin 0.8s linear infinite;}
      @keyframes spin{to{transform:rotate(360deg)}}
      p{color:#6b7280;font-size:14px;margin:0;}
      strong{color:#1f2937;font-size:16px;}
    </style></head>
    <body>
      <div class="spinner"></div>
      <strong>Smart Search</strong>
      <p>Starting AI engine…</p>
    </body></html>
  `));

  let actualPort = PORT;
  try {
    await waitForServer(DEFAULT_PORT, 60000);
    actualPort = getPort(); // read the port Python actually bound to
    console.log(`[Electron] Python backend ready on port ${actualPort}.`);
  } catch (err) {
    console.error('[Electron] Backend failed to start:', err.message);
    dialog.showErrorBox('Startup Error', `Smart Search backend failed to start.\n\n${err.message}`);
  } finally {
    loadingWin.close();
  }

  createWindow(actualPort);
  createTray(actualPort);

  // Show in Dock when window is visible
  if (process.platform === 'darwin') {
    app.dock.show();
  }
});

app.on('activate', () => {
  if (mainWindow) {
    mainWindow.show();
    mainWindow.focus();
  }
});

// Keep app alive when all windows are closed (live in tray)
app.on('window-all-closed', (e) => {
  if (process.platform !== 'darwin') {
    // On non-mac, allow normal quit
    app.isQuitting = true;
    app.quit();
  }
});

app.on('before-quit', () => {
  app.isQuitting = true;
  if (pythonProcess) {
    console.log('[Electron] Stopping Python backend…');
    pythonProcess.kill('SIGTERM');
  }
});
