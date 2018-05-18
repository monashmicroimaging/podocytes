# -*- mode: python -*-
import os.path
import PyInstaller.utils.hooks

spec_root = os.path.abspath(SPECPATH)

import gooey
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'),
                       prefix='gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix='gooey/images')

block_cipher = None

binaries = []
datas = []
datas += PyInstaller.utils.hooks.collect_data_files("bioformats")
datas += PyInstaller.utils.hooks.collect_data_files("javabridge")
datas += PyInstaller.utils.hooks.collect_data_files("skimage.io._plugins")
datas += [('podocytes/app-images/*', 'podocytes/app-images')]

hiddenimports = []
hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy.core")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("pandas")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy.special")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("wx")
hiddenimports += PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')

hiddenimports += [
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.skiplist',
    'scipy._lib.messagestream',
    'pywt._extensions._cwt'
]

a = Analysis(['bin/launch-gui.py'],
             pathex=[spec_root],  # add the path where the spec file is located
             binaries=binaries,
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

libpng_pathname = PyInstaller.utils.hooks.get_homebrew_path("libpng")
libpng_pathname = os.path.join(libpng_pathname, "lib", "libpng16.16.dylib")

java_pathname = os.path.join(os.environ["JAVA_HOME"], "jre/lib/server/libjvm.dylib")

a.binaries += [
    ("libpng16.16.dylib", libpng_pathname, "BINARY"),
    ("libjvm.dylib", java_pathname, "BINARY")
]

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
