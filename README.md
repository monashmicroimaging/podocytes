# podocytes

## For users
### Install the dependencies
* [Legacy Java SE 6 runtime](https://support.apple.com/kb/DL1572?locale=en_AU)
[](www.oracle.com/technetwork/java/javase/downloads/index.html)

### 

#### Notes:
* Currently, you must be connected to the internet to run the app.
Pims relies on the Bioformats [loci_tools.jar](http://downloads.openmicroscopy.org/bio-formats/) 
and downloads this at program runtime.

## For developers
### Setup development environment

```python
conda create -n podo environment.yml
```

### Running PyInstaller to create macOS build

```python
pip install -e .
pyinstaller Podo.spec -w -F -y
```
