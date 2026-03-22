'use strict';

const { contextBridge, ipcRenderer } = require('electron');

// Expose a safe API to the renderer (the Flask web page)
contextBridge.exposeInMainWorld('electron', {
  isElectron: true,

  // Opens the native macOS folder-picker dialog
  openFolderDialog: () => ipcRenderer.invoke('open-folder-dialog'),

  // Opens a URL in the default browser
  openUrl: (url) => ipcRenderer.invoke('open-url', url),
});
