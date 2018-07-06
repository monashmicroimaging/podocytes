# podocytes

## For users
### Install the dependencies
* [Legacy Java SE 6 runtime](https://support.apple.com/kb/DL1572?locale=en_AU)
[](www.oracle.com/technetwork/java/javase/downloads/index.html)

### Install Podo
* Download [Podo.app.zip](https://github.com/monashmicroimaging/podocytes/releases/tag/v0.1.0-alpha)
(currently available only for Mac) 

Unzip the file, then move the app to /Applications/

When opening the app for the first time, make sure to hold down **Control** while clicking on the app, and then select 'Open'. 
Then confirm you want to open this app from an unidentified developer. 
[See here](https://www.imore.com/how-open-apps-unidentified-developers-mac)
for step by step screenshots.


#### Notes:
* Currently, you must be connected to the internet to run the app.
Pims relies on the Bioformats [loci_tools.jar](http://downloads.openmicroscopy.org/bio-formats/) 
and downloads this at program runtime.
* The output directory location must not include any spaces in the path. Eg: `/Documents/path/to/output/` is fine, but `Documents/folder with spaces/to/output/` is not. 

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
