#!/usr/bin/env python
# Kept for compatibility: this tool is now part of the pony3d package and is
# installed as the `pony3d-breakcube` command. This wrapper runs it from a clone.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pony3d.tools.breakcube import main

if __name__ == '__main__':
    main()
