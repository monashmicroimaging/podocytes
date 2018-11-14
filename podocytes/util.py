import os
import time
import logging

from gooey.python_bindings.gooey_decorator import Gooey as gooey
from gooey.python_bindings.gooey_parser import GooeyParser

from podocytes.__init__ import __version__


def find_files(input_directory, ext):
    """Recursive search for filenames matching specified extension.

    Parameters
    ----------
    input_directory : str
        Filepath to input directory location.
    ext : str
        File extension (eg: '.tif', '.lif', etc.)

    Returns
    -------
    filelist : list of str
        List of files matching specified extension in the input directory path.
    """
    filelist = []
    for root, _, files in os.walk(input_directory):
        for f in files:
            # ignore hidden files
            if f.endswith(ext) and not f.startswith('.'):
                filename = os.path.join(root, f)
                filelist.append(filename)
    return filelist


def configure_parser_default(parser):
    parser.add_argument('input_directory', widget='DirChooser',
                        help='Folder containing files for processing.')
    parser.add_argument('output_directory', widget='DirChooser',
                        help='Folder to save output analysis files.')
    parser.add_argument('glomeruli_channel_number',
                        help='Fluorescence channel with glomeruli.',
                        type=int, default=1)
    parser.add_argument('podocyte_channel_number',
                        help='Fluorescence channel with podocytes.',
                        type=int, default=2)
    parser.add_argument('minimum_glomerular_diameter',
                        help='Minimum glomerular diameter (microns).',
                        type=float, default=30)
    parser.add_argument('maximum_glomerular_diameter',
                        help='Maximum glomerular diameter (microns).',
                        type=float, default=300)
    parser.add_argument('file_extension',
                        help='Extension of image file format (.tif, etc.)',
                        type=str, default='.lif')
    return parser


def log_file_begins(args):
    """Initialize logging and begin writing log file.

    Parameters
    ----------
    args : user input arguments
        Input arguments from user.

    Returns
    -------
    log_filename : str
        Filepath to output log text file location.
    """
    time_start = time.time()
    timestamp = time.strftime('%d-%b-%Y_%H-%M%p', time.localtime())
    log_filename = os.path.join(args.output_directory, f"log_podo_{timestamp}.log")
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{log_filename}"),
            logging.StreamHandler()
        ])
    # Log user input arguments
    logging.info(f"Podocyte automated analysis program, version {__version__}")
    logging.info(f"{timestamp}")
    logging.info("========== USER INPUT ARGUMENTS ==========")
    user_input = vars(args)
    for key, val in user_input.items():
        logging.info(f"{key}: {val}")
    logging.info("======= END OF USER INPUT ARGUMENTS =======")
    return time_start


def log_file_ends(time_start, total_gloms_counted=None):
    """Append runtime information to log.

    Parameters
    ----------
    time_start : datetime
        Datetime object from program start time.
    total_gloms_counted : int
        The number of glomeruli identified and analyzed.

    Returns
    -------
    time_delta : datetime.timedelta
        How long the program took to run.
    """
    time_end = time.time()
    time_delta = time_end - time_start
    minutes, seconds = divmod(time_delta, 60)
    logging.info(f'Total runtime: '
                 f'{round(time_delta)} seconds.')
    if total_gloms_counted:
        seconds_per_glom = time_delta / total_gloms_counted
        logging.info(f'Average time per glomerulus: '
                     f'{round(seconds_per_glom)} seconds.')
    logging.info('Program complete.')
    return time_delta
