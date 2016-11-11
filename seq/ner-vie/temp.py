#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs


f1 = codecs.open('out.txt', 'r', 'utf-8')
f2 = codecs.open('out_new.txt', 'w', 'utf-8')
for line in f1:
    if line.startswith('<editor>') or line.startswith('-DOCSTART-') or line.startswith('<s>') or line == u'\n' or line == u'﻿<title>Đời thuyền_viên (kỳ 3): Những người bỏ_xác giữa đại_dương.</title>':
        pass
    elif line.startswith('</s>'):
        f2.write(u'\n')
    else:
        f2.write(line)
f1.close()
f2.close()
