import numpy as np
from astropy.time import Time

## data format https://www.sws.bom.gov.au/World_Data_Centre/2/8/8
## data source https://www.sws.bom.gov.au/World_Data_Centre/1/9

def readchunk(chunk):
    rec_bytes = np.array([8, 8, 8, 8, 8, 501, 501, 501, 501])
    rec_pos_ed = np.cumsum(rec_bytes)
    rec_pos_bg = rec_pos_ed - rec_bytes

    i = 0
    recordhd = chunk[rec_pos_bg[i]:rec_pos_ed[i]]
    i += 1
    abandhd = chunk[rec_pos_bg[i]:rec_pos_ed[i]]
    i += 1
    bbandhd = chunk[rec_pos_bg[i]:rec_pos_ed[i]]
    i += 1
    cbandhd = chunk[rec_pos_bg[i]:rec_pos_ed[i]]
    i += 1
    dbandhd = chunk[rec_pos_bg[i]:rec_pos_ed[i]]
    i += 1
    abanddata = chunk[rec_pos_bg[i]:rec_pos_ed[i]]
    i += 1
    bbanddata = chunk[rec_pos_bg[i]:rec_pos_ed[i]]
    i += 1
    cbanddata = chunk[rec_pos_bg[i]:rec_pos_ed[i]]
    i += 1
    dbanddata = chunk[rec_pos_bg[i]:rec_pos_ed[i]]

    yy = recordhd[0]
    mm = recordhd[1]
    dd = recordhd[2]
    h = recordhd[3]
    m = recordhd[4]
    s = recordhd[5]
    dataok = recordhd[6]
    unused = recordhd[7]

    tstr = '20{:02d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}'.format(yy, mm, dd, h, m, s)
    # sfreqs = [[],[],[],[]]
    # efreqs = [[],[],[],[]]
    # fresbws = [[],[],[],[]]
    # reflvls = [[],[],[],[]]
    # ranlvls = [[],[],[],[]]
    # for bdidx,bandhd in enumerate([abandhd,bbandhd,cbandhd,dbandhd]):
    #     for l in bandhd[0:2]:
    #         sfreqs[bdidx].append(l)
    #     for l in bandhd[2:4]:
    #         efreqs[bdidx].append(l)
    #     for l in bandhd[4:6]:
    #         fresbws[bdidx].append(l)
    #     reflvls[bdidx].append(bandhd[6])
    #     ranlvls[bdidx].append(bandhd[7])

    spec = [[], [], [], []]

    for bdidx, bddata in enumerate([abanddata, bbanddata, cbanddata, dbanddata]):
        for l in bddata:
            spec[bdidx].append(l)

    spec = np.hstack(spec)
    return spec, tstr, dataok

