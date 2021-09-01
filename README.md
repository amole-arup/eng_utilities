# eng_utilities
## General Engineering Tools Repository

I am currently working out how to set up a project, so it may take a little while for it to be user-friendly.

However, the following two functions should work:

```python
from eng_utilities.E2K_parsing import run_all
from eng_utilities.GWA_utilities import write_GWA

E2K_path = r'C:\myfolder\myE2Kfile.E2K'
E2K_dict = run_all(E2K_path)

GWA_path = r'C:\myfolder\myE2Kfile.GWA'
write_GSA(E2K_dict, GWA_path)
```

The E2K_dict file contains some summaries of quantities in `E2K_dict[MEMBERS SUMMARY]` and `E2K_dict[SHELLS SUMMARY]`

## Notes & Warnings:
- it does not do OFFSETYI etc - it may need a coordinate system to do this...
- it does axial offsets & cardinal points - but this should be reviewed
- it does not do rigid offset to make beams more flexible
- it does not do loads
- it cannot take account of property modifications for individual members (PROPMODS)
- does not address line constraints

- it does not do piers and spandrels (although that may be possible using GSA Assemblies
- Not sure what to do with LINE and AREA which are not beams, columns, braces, floors, panels or ramps
- handling of asymmetric sections (such as angles / L-sections) has not been rigorously addressed

## TODO: 
    1. Review the aggregation of 2D element areas, volumes and weights   
    2. Add loads to the summary
    3. Look at incorporating the tools for writing to GSA (needs offsets, etc)
    4. Look at including 'Towers' definition
    5. Modifiers???
    6. Element centroids and beam and element normals (loads can be located to generate global moments)

