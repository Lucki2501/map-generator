# map-generator
Python map generator using numpy and openCV

## Steps to run this file
- Replace the `PATH` constant with the directory you want your images to be saved in.
- Run the file!

The only function to run is create_map(), which takes the following arguments:
- `template`: One of the keys from the `map_types` array. Defaults to `'normal'`.
- `size`: One of the keys from the `sizes` array. Defaults to `'1024x1024'`.
- `cities`: Whether cities should be generated. Defaults to `True`.
- `seed`: The seed the map's random generation is computed with. If none is set, it will be a random integer between 1 and 10000. Defaults to `None`.
