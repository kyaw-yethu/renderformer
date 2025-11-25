# RenderFormer Dataset README

## Requirements
1. pip install -r requirements.txt
2. prepare object mesh files or using `.obj` files in examples/objects/ 

## How to use
```bash

python scripts/generate_dataset.py \
    --config scripts/configs/large.json
```

## Parameters
- `num_scenes` (int): Number of scenes to generate
- `num_views_per_scene` (int): Number of views per scene
- `template_id` (int | [int, int] | null): Template ID
  - `null`: Random (0-3)
  - `int`: Fixed value (e.g., `0`)
  - `[min, max]`: Random within range (e.g., `[0, 3]`)
- `num_objects` (int | [int, int] | null): Number of objects per scene
  - `null`: Random (1-3 objects, 50/30/20 ratio)
  - `int`: Fixed value
  - `[min, max]`: Random within range
- `num_lights` (int | [int, int] | null): Number of lights per scene
  - `null`: Random (2-4)
  - `int`: Fixed value
  - `[min, max]`: Random within range

### Scene Space Settings
- `scene_center` ([float, float, float]): Scene center coordinates (Default: `[0.0, 0.0, 0.0]`)
- `scene_bounds` ([float, float]): Scene boundary range (Default: `[-0.5, 0.5]`)
- `ground_level` (float): Ground height (Default: `-0.5`)

### Camera Settings
- `camera_distance_range` ([float, float]): Camera distance range (Default: `[1.5, 2.0]`)
- `camera_fov_range` ([float, float]): Camera field of view (FOV) range (Default: `[30.0, 60.0]`)

### Light Settings
- `light_distance_range` ([float, float]): Light distance range (Default: `[2.1, 2.7]`)
- `light_scale_range` ([float, float]): Light size range (Default: `[2.0, 2.5]`)
- `light_emission_range` ([float, float]): Light emission intensity range (Default: `[2500.0, 5000.0]`)

### Rendering Settings
- `resolution` (int): Rendering resolution (Default: `256`, optimized for RTX 3090)
- `spp` (int): Samples per pixel (Default: `512`)

### Object Mesh Settings
- `max_triangles` (int): Maximum number of triangles per object (Default: `2048`)
- `min_triangles` (int): Minimum number of triangles per object (Default: `100`)
- `objaverse_mesh_paths` (string[] | null): List of object mesh file paths to use
  - `null`: No objects (objects will not be rendered)
  - `string[]`: Array of object file paths (e.g., `[“examples/objects/shader-ball/ball.obj”, ...]`)
- `objaverse_root` (string | null): Objaverse root directory (Currently using in examples not using this options.) 

### Output and Execution Settings
- `examples_dir` (string): Example directory path (default: `“examples”`)
- `output_dir` (string): Dataset output directory
- `num_workers` (int | null): Number of parallel workers (`null` means auto-determined)
- `skip_rendering` (bool): Skip GT rendering (default: `false`)
- `gpu_memory_check` (bool): Enable GPU memory check (default: `true`)
- `base_seed` (int | null): Random seed (default: `42`)

## Generated Dataset Structure
```
output/dataset_dir/
├── metadata.json          # Metadata (scene info, paths, etc.)
├── scenes/                # Scene JSON files
│   ├── scene_0.json
│   ├── scene_1.json
│   └── ...
├── h5/                    # HDF5 data files
│   ├── scene_0.h5
│   ├── scene_1.h5
│   └── ...
├── renders/               # Rendered images (GT)
│   ├── scene_0_view_0.exr
│   ├── scene_0_view_1.exr
│   └── ...
└── meshes/                # Mesh files
    └── split/
        ├── background_0.obj
        └── ...
```

## Dataset Creation Steps
1. Scene Creation
   - Create template background
   - Object placement (if objaverse_mesh_paths provided)
   - Light placement
   - Camera placement

2. JSON → HDF5 Conversion
   - Convert scene information to HDF5 format
   - Save triangles, textures, vertex normals, etc.

3. GT Rendering (skip when skip_rendering=True)
   - Render ground truth images using Blender
   - Requires **bpy module**

4. Metadata Generation (Step 4)
   - Save all scene information to metadata.json

## Rendering with Renderformer
Inference using the generated H5 file works the same as with Renderformer.
Example:

```bash
python infer.py \
    --h5_file output/my_dataset/h5/scene_0.h5 \
    --resolution 256 \
    --precision fp16 \
    --output_dir output/renderformer_output \
    --tone_mapper pbr_neutral
```