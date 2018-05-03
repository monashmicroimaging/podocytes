import sys
from . import lifio
from skimage import io

def main():
    fin, fout = sys.argv[1:]
    image = lifio.read_image_series(fin)
    io.imsave(fout, image)
    lifio.done()  # kill javabridge, or python won't quit

if __name__ == '__main__':
    main()
