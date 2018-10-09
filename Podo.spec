# -*- mode: python -*-
import os.path
import PyInstaller.utils.hooks

spec_root = os.path.abspath(SPECPATH)

# Get gooey data files
import gooey
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'),
                       prefix='gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix='gooey/images')


# Get loci_tools.jar; ensure it's packaged in a.binaries, below
from pims import bioformats
loci_jar_full_path = bioformats._download_jar(version='5.9.2')
loci_dir, loci_file = os.path.split(loci_jar_full_path)

block_cipher = None

binaries = []

datas = []
datas += PyInstaller.utils.hooks.collect_data_files("skimage.io._plugins")
datas += [('podocytes/app-images/*', 'podocytes/app-images')]

hiddenimports = []
hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy.core")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("pandas")
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

excludes = []
excludes += [
    "SimpleITK",
    "pyamg",
    "sphinx",
    "whoosh",
    "glib",
    "PyQt5.QtGui",
    "PyQt5.QtCore",
    "PyQt4.QtGui",
    "PyQt4.QtCore",
    "PySide.QtGui",
    "PySide.QtCore",
    "astropy",
    "PyQt5",
    "PyQt4",
    "PySide",
    "PySide2",
    "gtk",
    "FixTk",
    "tcl",
    "tk",
    "_tkinter",
    "tkinter",
    "Tkinter"
]

a = Analysis(['bin/launch-gui.py'],
             pathex=[spec_root],  # add the path where the spec file is located
             binaries=binaries,
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=excludes,
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

libpng_pathname = PyInstaller.utils.hooks.get_homebrew_path("libpng")
libpng_pathname = os.path.join(libpng_pathname, "lib", "libpng16.16.dylib")

java_pathname = os.path.join(os.environ["JAVA_HOME"], "jre/lib/server/libjvm.dylib")

a.binaries += [
    ("libpng16.16.dylib", libpng_pathname, "BINARY"),
    ("libjvm.dylib", java_pathname, "BINARY"),
    (loci_file, loci_dir, "BINARY")
]

exclude_binaries = [
    ('libpng16.16.dylib', '/usr/local/lib/python2.7/site-packages/matplotlib/.dylibs/libpng16.16.dylib', 'BINARY'),
    ('libwx_osx_cocoau_webview-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_webview-3.0.dylib', 'BINARY'),
    ('libwx_osx_cocoau_html-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_html-3.0.dylib', 'BINARY'),
    ('libwx_osx_cocoau_xrc-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_xrc-3.0.dylib', 'BINARY'),
    ('libwx_osx_cocoau_core-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_core-3.0.dylib', 'BINARY'),
    ('libwx_osx_cocoau_adv-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_adv-3.0.dylib', 'BINARY'),
    ('libwx_osx_cocoau_qa-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_qa-3.0.dylib', 'BINARY'),
    ('libwx_baseu_xml-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_baseu_xml-3.0.dylib', 'BINARY'),
    ('libwx_baseu_net-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_baseu_net-3.0.dylib', 'BINARY'),
    ('libwx_baseu-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_baseu-3.0.dylib', 'BINARY'),
    ('libwx_osx_cocoau_stc-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_stc-3.0.dylib', 'BINARY')
]

a.binaries = [binary for binary in a.binaries if binary not in exclude_binaries]

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
