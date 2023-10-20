import sys
from pathlib import Path
path_root = Path(__file__).parent.parent
sys.path.append(str(path_root))
print(sys.path)
from source import pqEDMD

duff_EDMD = pqEDMD(p=[8], q=[0.5, 1],
                   polynomial='Legendre',
                   normalization=True,
                   method="")
print(duff_EDMD)
