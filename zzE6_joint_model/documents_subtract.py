import codecs
import os

from config import cfg

cfg.init()

def load_class_list(path,  encoding='utf-8'):
    if path is not None:
        return codecs.open(path, 'r', encoding=encoding).read().split('\n')
    else:
        return None

def save_class_list(class_list, path, encoding='utf-8'):
    # type: (list, basestring) -> None
    f = codecs.open(path, 'w', encoding=encoding)
    for cls in class_list:
        f.write(unicode(cls) + '\n')
    f.seek(-1, os.SEEK_END)
    f.truncate()
    f.close()



doc_a = load_class_list('class_keep_from_pruning.txt')
doc_b = load_class_list('class_keep_from_pruning-train.txt')
doc_c_name = 'class_keep_from_pruning-test-2.txt'
doc_c = []

for line in doc_a:
    if line not in doc_b:
        doc_c.append(line)

save_class_list(doc_c, doc_c_name)
