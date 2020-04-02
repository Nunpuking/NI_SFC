from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import collections
import numpy as np

def time_format(s):
    h = math.floor(s / 3600)
    m = math.floor((s-3600*h) / 60)
    s = s - h*3600 - m*60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (time_format(s))

