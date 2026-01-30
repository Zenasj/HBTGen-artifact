# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

DllFiles = [
         (r'C:\Users\\PC-2001-044\anaconda3\pkgs\pytorch-1.7.1-py3.8_cuda101_cudnn7_0\Lib\site-packages\torch\lib\*.dll','.'),
	 ]


a = Analysis(['main.py'],
             pathex=[r'C:\Users\PC-2001-044\Desktop\アノマリー検出\210129実装テスト\ExecFileCreation\2. PaDiM\8. OneFileMerge'],
             binaries=DllFiles,
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )