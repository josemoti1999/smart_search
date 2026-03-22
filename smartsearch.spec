# smartsearch.spec  —  PyInstaller spec for the Smart Search Flask backend
# Usage: pyinstaller smartsearch.spec
# Output: dist/smartsearch  (~150 MB standalone binary)

block_cipher = None

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

fastembed_datas = collect_data_files('fastembed')
pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[] + pil_binaries,
    datas=[
        ('templates', 'templates'),
        ('memory_helper.py', '.'),
        ('config.py', '.'),
        ('embedder.py', '.'),
    ] + fastembed_datas + pil_datas,
    hiddenimports=[
        'flask', 'werkzeug', 'jinja2', 'click', 'itsdangerous',
        'fastembed', 'fastembed.text', 'onnxruntime',
        'faiss',
        'PyPDF2', 'docx',
        'groq', 'rank_bm25', 'psutil',
        'dotenv',
        'numpy',
    ] + collect_submodules('fastembed')
      + pil_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'sentence_transformers',
        'bitsandbytes', 'accelerate',
        'google.generativeai',
        'IPython', 'matplotlib', 'cv2',
        'tkinter', '_tkinter',
        'tensorflow', 'keras',
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='smartsearch',
    debug=False,
    strip=False,
    upx=True,
    console=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
