# LEVIR-CD

We have supplemented geospatial information (e.g., latitude and longitude coordinates) for each sample in LEVIR_CD. Specifically, we provide a `.geojson` file and a `.json` file, both of which contains the correspondence of the sample name and the geospatial location. Files can be seen in the `LEVIR_CD_spatial_info` folder.

Here, we give more details about these files.

## geojson file

```json
{"type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            -97.999417, 30.172574  // the longitude and latitude of the left up point of one sample image. 
          ],
          [
            -97.999417, 30.167081  // the longitude and latitude of the right up point of one sample image. 
          ],
          [
            -97.993924, 30.167081  // the longitude and latitude of the right bottom point of one sample image. 
          ],
          [
            -97.993924, 30.172574  // the longitude and latitude of the left bottom point of one sample image. 
          ],
          [
            -97.999417, 30.172574  // the longitude and latitude of the left up point of one sample image. 
          ]
        ]
      },
      "properties": {
        "name": "train_1.png"  // the sample name
      }
    },
    ...
   ]
 }
```



## json file

```json
{"train_1.png": [-97.99941748380661, 30.17257422208786, -97.99392431974411, 30.16708105802536], // sample name: [the longitude, latitude of the left up point, the longitude, latitude of the right bottom point.]
 ...
}
```

