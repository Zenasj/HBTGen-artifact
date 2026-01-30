import os
import tempfile
import zipfile


def compress_debug(zipname):
    # find debug thing
    with zipfile.ZipFile(zipname, 'r') as orig:
        for name in orig.namelist():
            if "debug" in name:
                debug_filename = name
                data = orig.read(name)

    # generate a temp file
    tmpfd, tmpname = tempfile.mkstemp(dir=os.path.dirname(zipname))
    os.close(tmpfd)

    # create a temp copy of the archive without debug info
    with zipfile.ZipFile(zipname, 'r') as zin:
        with zipfile.ZipFile(tmpname, 'w') as zout:
            for item in zin.infolist():
                if item.filename != debug_filename:
                    zout.writestr(item, zin.read(item.filename))

    # replace with the temp archive
    os.remove(zipname)
    os.rename(tmpname, zipname)

    # now add debug info with compression
    with zipfile.ZipFile(zipname, mode='a', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(debug_filename, data)


compress_debug("blaze.zip")