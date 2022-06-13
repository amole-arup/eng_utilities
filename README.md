# eng_utilities
## General Engineering Tools Repository

I am currently working out how to set up a project, so it may take a little while for it to be user-friendly.

However, the following two functions should work:

```python
from eng_utilities.E2K_parsing import run_all
from eng_utilities.GWA_utilities import write_GWA

# Generate a dictionary of dictionaries of dictionaries (and some lists)
# that contain the model information (with some post-processing)
E2K_path = r'C:\myfolder\myE2Kfile.E2K'
E2K_dict = run_all(E2K_path)

# Export a GWA text file for import into GSA
GWA_path = r'C:\myfolder\myE2Kfile.GWA'
write_GWA(E2K_dict, GWA_path)
```

The E2K_dict file contains some summaries of quantities in `E2K_dict['MEMBERS SUMMARY']` and `E2K_dict['SHELLS SUMMARY']`

## Notes & Warnings:

### E2K Parsing
- This generates a dictionary of dictionaries (of dictionaries) that is largely structured in the same way as the E2K file.
- Most sub-dictionaries are post-processed to provide additional structure (often for simplifying export to other packages)
- It should be possible to export the python dictionary straight to JSON


### GSA Export
- Exports materials to advanced material properties (but not to code materials)
- Exports both to 'analysis layer' and to the 'design layer'
- Intersecting beams are split in the analysis layer
- Meshing of large polygonal slabs can be done in the 'design layer'
- It does axial offsets, OFFSETYI etc & cardinal points - but it does not take account of coordinate systems and this should be reviewed
- It does not do rigid offset to make beams more flexible
- It does most point, beam and area loads, but not diaphragm loads (no wind, seismic)
- Property modifications have not been implemented in writing to the GWA file
- It cannot take account of property modifications for individual members (PROPMODS) since GSA does not do this
- It does not address line constraints
- Piers and spandrels are not converted into GSA assemblies
- Not sure what to do with LINE and AREA which are not beams, columns, braces, floors, panels or ramps
- Handling of asymmetric sections (such as angles / L-sections) has not been rigorously addressed

## TODO: 
    1. Review the aggregation of 2D element areas, volumes and weights   
    2. Add loads to the summary
    3. Look at adding to the tools for writing to GSA (needs offsets, etc)
    4. Look at including 'Towers' definition
    5. Modifiers???
    6. Element centroids and beam and element normals (loads can be located to generate global moments)
    7. Add wind loads
    8. Add export to Speckle
    9. Add export to the design layer

