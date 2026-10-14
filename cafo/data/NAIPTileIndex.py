"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import math

import numpy as np
import pystac_client
import shapely.geometry


class NAIPTileIndex:
    """Utility class for performing NAIP tile lookups by location with the Microsoft
    Planetary Computer STAC API.

    Tile footprints are requested one `CELL_SIZE` x `CELL_SIZE` degree cell at a time
    and cached in memory, so repeated lookups in the same area only query the API once.
    """

    STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION = "naip"
    CELL_SIZE = 1.0

    def __init__(self):
        self.catalog = pystac_client.Client.open(NAIPTileIndex.STAC_API_URL)
        self.cells = {}

    def _get_cell(self, x, y):
        """Returns the URLs, footprints, and footprint bounds of all NAIP tiles that
        intersect the cell containing the (`x`, `y`) point.
        """
        key = (
            math.floor(x / NAIPTileIndex.CELL_SIZE),
            math.floor(y / NAIPTileIndex.CELL_SIZE),
        )
        if key not in self.cells:
            minx = key[0] * NAIPTileIndex.CELL_SIZE
            miny = key[1] * NAIPTileIndex.CELL_SIZE
            search = self.catalog.search(
                collections=[NAIPTileIndex.COLLECTION],
                bbox=[
                    minx,
                    miny,
                    minx + NAIPTileIndex.CELL_SIZE,
                    miny + NAIPTileIndex.CELL_SIZE,
                ],
                limit=1000,
            )
            urls = []
            shapes = []
            for item in search.items_as_dicts():
                urls.append(item["assets"]["image"]["href"])
                shapes.append(shapely.geometry.shape(item["geometry"]))
            bounds = np.array([shape.bounds for shape in shapes]).reshape(-1, 4)
            self.cells[key] = (urls, shapes, bounds)
        return self.cells[key]

    def lookup_point(self, lat, lon):
        """Given a lat/lon pair, return the list of NAIP tiles that *contain* that point.

        Args:
            lat (float): Latitude in EPSG:4326
            lon (float): Longitude in EPSG:4326

        Returns:
            intersected_files (list): A list of URLs of NAIP tiles that *contain* the
                given (`lat`, `lon`) point

        Raises:
            IndexError: Raised if no tile within the index contains the given
                (`lat`, `lon`) point
        """

        point = shapely.geometry.Point(float(lon), float(lat))
        geom = shapely.geometry.mapping(point)

        return self.lookup_geom(geom)

    def lookup_geom(self, geom):
        """Given a GeoJSON geometry, return the list of NAIP tiles that *contain* that feature.

        Args:
            geom (dict): A GeoJSON geometry in EPSG:4326

        Returns:
            intersected_files (list): A list of URLs of NAIP tiles that *contain* the
                given `geom`

        Raises:
            IndexError: Raised if no tile within the index fully contains the given `geom`
        """
        shape = shapely.geometry.shape(geom)

        # Any tile that contains `shape` also contains this point, so it intersects the
        # point's cell
        point = shape.representative_point()
        urls, shapes, bounds = self._get_cell(point.x, point.y)

        minx, miny, maxx, maxy = shape.bounds
        intersected_indices = np.flatnonzero(
            (bounds[:, 0] <= maxx)
            & (bounds[:, 1] <= maxy)
            & (bounds[:, 2] >= minx)
            & (bounds[:, 3] >= miny)
        )

        intersected_files = [
            urls[idx] for idx in intersected_indices if shapes[idx].contains(shape)
        ]

        if len(intersected_files) <= 0 and len(intersected_indices) > 0:
            raise IndexError(
                "There are overlaps with tile index, but no tile contains the shape"
            )
        elif len(intersected_files) <= 0:
            raise IndexError("No tile intersections")
        else:
            return intersected_files
