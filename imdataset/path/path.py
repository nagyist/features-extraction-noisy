import ntpath
import os

def path_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def lsdir(path, absolute=False, sortByName=True):
    dirs = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) == False)]
    if sortByName:
        dirs.sort()

    if absolute:
        nd = []
        for d in dirs:
            nd.append(os.path.join(path, d))
        dirs = nd

    return dirs

def lsfile_all(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def lsfile(path, absolute=False, extensionList=None, sortByName=True, listHidden=False):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    rightFiles=[]
    for f in files:
        if listHidden==False and f.startswith("."): continue
        if extensionList is None:
            rightFiles.append(f)
        else:
            for e in extensionList:
                if f.lower().endswith(e.lower()):
                    rightFiles.append(f)
                    break
    if sortByName:
        rightFiles.sort()

    if absolute:
        rf = []
        for f in rightFiles:
            rf.append(os.path.join(path, f))
        rightFiles = rf

    return rightFiles




DEFAULT_EXTENSIONS=[".jpg", ".jpeg", ".png", ".gif", ".bmp"]

import imdataset
def lsim(path, absolute_path=False, img_limit=-1, img_skip=0, extensions=DEFAULT_EXTENSIONS, sort_by_name=True):
    imgFiles = lsfile(path, absolute_path, extensions, sort_by_name, listHidden=False)

    f, l, n = imdataset.utils.subsequence_index(img_skip, img_limit, len(imgFiles))
    imgFiles = imgFiles[f:l]
    return imgFiles