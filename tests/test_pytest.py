import eng_utilities
import pytest

#def test_E2K

def test_always_passes():
    assert True


def test_always_fails():
    assert False


# ============================
# =======  E2K_parsing =======
# ============================

# >= 2013
"""
$ LOAD PATTERNS
$ POINT OBJECT LOADS
$ FRAME OBJECT LOADS
$ SHELL OBJECT LOADS
$ ANALYSIS OPTIONS
$ MASS SOURCE
$ FUNCTIONS
$ GENERALIZED DISPLACEMENTS
$ LOAD CASES
$ LOAD COMBINATIONS
"""

new_lp = """$ LOAD PATTERNS
  LOADPATTERN "SW"  TYPE  "Dead"  SELFWEIGHT  1
  LOADPATTERN "SDL1"  TYPE  "Super Dead"  SELFWEIGHT  0
  LOADPATTERN "LL2"  TYPE  "Live"  SELFWEIGHT  0
  LOADPATTERN "WX"  TYPE  "Wind"  SELFWEIGHT  0
  LOADPATTERN "EQX"  TYPE  "Seismic"  SELFWEIGHT  0
  LOADPATTERN "BLAST"  TYPE  "Other"  SELFWEIGHT  0
"""

new_lc = """$ LOAD CASES
  LOADCASE "Modal"  TYPE  "Modal - Ritz"  INITCOND  "PRESET"  
  LOADCASE "Modal"  ACCEL  "UX"  
  LOADCASE "Modal"  ACCEL  "UY"  
  LOADCASE "Modal"  ACCEL  "UZ"  
  LOADCASE "Modal"  MAXMODES  30 MINMODES  30 
  LOADCASE "SW"  TYPE  "Linear Static"  INITCOND  "PRESET"  
  LOADCASE "SW"  LOADPAT  "SW"  SF  1 
  LOADCASE "WX"  TYPE  "Linear Static"  INITCOND  "PRESET"  
  LOADCASE "WX"  LOADPAT  "WX"  SF  1 
"""

new_comb = """$ LOAD COMBINATIONS
  COMBO "SDL"  TYPE "Linear Add"  
  COMBO "SDL"  LOADCASE "SDL1"  SF 1 
  COMBO "SDL"  LOADCASE "SDL2"  SF 1 
"""


# <=9.7
"""
$ STATIC LOADS
$ POINT OBJECT LOADS
$ AREA OBJECT LOADS
$ ANALYSIS OPTIONS
$ FUNCTIONS
$ RESPONSE SPECTRUM CASES
$ LOAD COMBINATIONS
"""

old_static = """$ STATIC LOADS
  LOADCASE "DEAD"  TYPE  "DEAD"  SELFWEIGHT  1
  LOADCASE "LIVE"  TYPE  "LIVE"  SELFWEIGHT  0
  LOADCASE "SDL"  TYPE  "DEAD"  SELFWEIGHT  0
  SEISMIC "EQXBW"  "UBC97"    DIR "X"  TOPSTORY "L66"    BOTTOMSTORY "BASE"   PERIODTYPE "PROGCALC"   CT 0.035  SOIL "SC"  Z 0.4  SOURCETYPE "B"  SOURCEDIST 2  I 1  R 4.5
  WIND "NBCCX"  "NBCC2005"  EXPOSUREFROM "DIAPHRAGMS"    ANG 0  WINDWARDCP 0.8  LEEWARDCP 0.5  TOPSTORY "L60"  BOTTOMSTORY "L1"  VELOCITYPRESSURE 2  GUSTFACTOR 2  IW 1
  WIND "NBCCX1"  "USER"  
  WIND  "NBCCX1"  USERLOAD  "L66"  "D1"  FX 1004.32  XLOC 16.5  YLOC 27  
"""

old_combo = """
$ LOAD COMBINATIONS
  COMBO "COMB1"  TYPE "ADD" DESIGN "CONCRETE"
  COMBO "COMB1"  LOAD "DEAD"  SF 1.2
  COMBO "COMB1"  LOAD "SDL"  SF 1.2
  COMBO "COMB1"  LOAD "LIVE"  SF 1
  COMBO "COMB1"  LOAD "ASCEX"  SF 1.3
  COMBO "DL"  TYPE "ADD"  
  COMBO "DL"  LOAD "DEAD"  SF 1
  COMBO "DL"  LOAD "SDL"  SF 1
"""


from E2K_parsing import load_parser, gather, line_split

pt_ld1 = """$ POINT OBJECT LOADS
  POINTLOAD  "4357"  "L008"  TYPE "FORCE"  LC "BLAST"    FZ -1950
  POINTLOAD  "4357"  "L008"  TYPE "FORCE"  LC "SW"    FZ -240.2  MX -613
  POINTLOAD  "4357"  "L008"  TYPE "FORCE"  LC "SDL"    FZ -97.02  MX -268.9
"""

ln_ld1 = """$ FRAME OBJECT LOADS
  LINELOAD  "C19"  "L024"  TYPE "POINTF"  DIR "1"  LC "TEST"  FVAL -1000  RDIST 0
  LINELOAD  "B380"  "L003"  TYPE "UNIFF"  DIR "GRAV"  LC "SDLF"  FVAL 3.2
  LINELOAD  "B1897"  "L043"  TYPE "TRAPF"  DIR "GRAV"  LC "SDLF"  FSTART 4.3  FEND 4.3  RDSTART 0  RDEND 0.5  
  LINELOAD  "B1897"  "L043"  TYPE "TRAPF"  DIR "GRAV"  LC "SDLF"  FSTART 4.3  FEND 4.3  RDSTART 0.5  RDEND 1  
  LINELOAD  "B383"  "L040"  TYPE "UNIFF"  DIR "GRAV"  LC "SDLF"  FVAL 4.3
  LINELOAD  "B5721"  "L040"  TYPE "POINTF"  DIR "GRAV"  LC "SW"  FVAL 95  RDIST 0.25
"""

sh_ld1 = """$ SHELL OBJECT LOADS
  AREALOAD  "F566"  "BM2"  TYPE "UNIFF"  DIR "GRAV"  LC "SDL"  FVAL 1.6
  AREALOAD  "F566"  "BM2"  TYPE "UNIFF"  DIR "GRAV"  LC "LL2"  FVAL 1.6
"""


def test_load_parser():
    assert ([load_parser(dict(tuple(gather(line_split(line))))) for line in ln_ld1.split('\n')[1:4] if len(line) > 0] == 
    [[{'TYPE': 'POINTF', 'DIR': 1, 'DATA': (0, -1000)}],
        [{'TYPE': 'UNIFF',
        'DIR': 'GRAV',
        'DATA': ((0, 3.2), (1, 3.2)),
        'AVE_LOAD': 3.2}],
        [{'TYPE': 'TRAPF',
        'DIR': 'GRAV',
        'DATA': ((0, 4.3), (0.5, 4.3)),
        'AVE_LOAD': 2.15}],
        [{'TYPE': 'TRAPF',
        'DIR': 'GRAV',
        'DATA': ((0.5, 4.3), (1, 4.3)),
        'AVE_LOAD': 2.15}],
        [{'TYPE': 'UNIFF',
        'DIR': 'GRAV',
        'DATA': ((0, 4.3), (1, 4.3)),
        'AVE_LOAD': 4.3}],
        [{'TYPE': 'POINTF', 'DIR': 'GRAV', 'DATA': (0.25, 95)}]])



