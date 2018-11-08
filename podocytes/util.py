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
