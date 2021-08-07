
The following data files are included in this directory:
- `chesapeake-bay-county-geoids.csv` - A listing of county GEOIDS that intersect with the Chesapeake Bay Watershed.
- `delmarva_all_set_tiles.geojson` - A spatial index of all NAIP imagery tiles that intersect with the extent of the Soroka and Duren dataset.
- `delmarva_testing_set_polygons.geojson` - The set of polygons from the Soroka and Duren dataset that fall within the testing split area.
- `delmarva_testing_set_tiles.geojson` - The set of NAIP tiles that fall within the testing split area.
- `delmarva_training_set_polygons.geojson` - The set of polygons from the Soroka and Duren dataset that fall within the training split area *and that are not intersected by any of the NAIP tiles in the test set*.
- `delmarva_training_set_tiles.geojson` - The set of NAIP tiles that fall within the training split area.
- `qualitative_eval_tile.csv` - A 2018 NAIP tile from Virginia that contains a large number of poultry barns.

See [agcensus_data/README.md](agcensus_data/README.md) and [poultry_barn_change_predictions/README.md](poultry_barn_change_predictions/README.md) for more information about the data in the subdirectories.