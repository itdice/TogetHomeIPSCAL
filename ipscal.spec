# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['ipscal.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
            'eventlet.hubs.epolls',
            'eventlet.hubs.kqueue',
            'eventlet.hubs.selects',
            'dns',
            'dns.dnssec',
            'dns.e164',
            'dns.hash',
            'dns.namedict',
            'dns.tsigkeyring',
            'dns.update',
            'dns.version',
            'dns.zone',
            'dns.asyncbackend',
            'dns.asyncquery',
            'dns.asyncresolver',
            'dns.versioned'
         ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ipscal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['.w64.icon'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ipscal',
)
