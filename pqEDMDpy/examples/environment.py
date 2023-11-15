from pqEDMDpy import pqEDMD

duff_EDMD = pqEDMD(p=[8], q=[0.5, 1],
                   polynomial='Legendre',
                   normalization=True,
                   method="")
print(duff_EDMD)
