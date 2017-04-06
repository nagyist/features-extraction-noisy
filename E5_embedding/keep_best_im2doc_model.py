import os
from config import cfg

WORKING_DIR = '/mnt/das4-fs4/var/node436/local/rdchiaro/features-extraction'
OUTPUT_DIR = '/mnt/das4-fs4/var/node436/local/rdchiaro/features-extraction/scremo'


os.chdir(WORKING_DIR)
files = os.listdir('./')



class WeightFile:
    def __init__(self, filename):
        self.filename = filename
        self.weight_n = int(filename.split('.weights.')[1].split('.')[0])
        self.key = filename.split('.weights.')[0]

file_containers = {}
for file in files:
    if file.endswith('.h5'):
        wfile = WeightFile(file)

        if wfile.key in file_containers.keys():
            file_containers[wfile.key].append(wfile)
        else:
            file_containers[wfile.key] = []
            file_containers[wfile.key].append(wfile)


if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


import shutil

for wflist in file_containers.values():
    best = wflist[0]
    for wfile in wflist:
        if wfile.weight_n > best.weight_n:
            best = wfile
    print("")
    print( "Best: " + best.filename)
    wflist.remove(best)
    print( "Will delete these files: " + str([ f.filename for f in wflist]))
    shutil.copy(best.filename, os.path.join(OUTPUT_DIR, best.filename))
