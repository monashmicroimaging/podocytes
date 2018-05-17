# -*- mode: python -*-
import gooey
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'),
                       prefix='gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix='gooey/images')

block_cipher = None

a = Analysis(['bin/launch-gui.py'],
             pathex=['/Users/jni/projects/podocytes'],
             binaries=[],
             datas=[('podocytes/app-images/*', 'podocytes/app-images')],
             hiddenimports=['pandas._libs.tslibs.timedeltas',
                            'pandas._libs.tslibs.np_datetime',
                            'pandas._libs.tslibs.nattype',
                            'pandas._libs.skiplist',
                            'scipy._lib.messagestream',
                            'pywt._extensions._cwt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

options = [('u', None, 'OPTION')]
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          options,
          gooey_languages,
          gooey_images,
          name='Podo',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False)
app = BUNDLE(exe,
             name='Podo.app',
             icon=None,
             bundle_identifier=None)
