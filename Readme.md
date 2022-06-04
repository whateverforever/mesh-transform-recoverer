# mesh-transform-recoverer

> Tool that recovers the transforms that were applied to a base "referencemesh"
> from copies of that mesh. E.g. the result of Michael Fogleman's [pack3d](https://www.michaelfogleman.com/pack3d/)

```
$ ./transform_recoverer.py --help                                                                                                            
usage: transform_recoverer.py [-h] [--outcsv OUTCSV] [--verify]
                              [--debug | --quiet]
                              referencemesh groupmesh

Tool that recovers the transforms that were applied to a base "referencemesh"
from copies of that mesh. E.g. the result of Michael Fogleman's `pack3d`.

positional arguments:
  referencemesh         Path to the mesh file for the reference object
  groupmesh             Path to the mesh that contains transformed instances
                        of `referencemesh`

optional arguments:
  -h, --help            show this help message and exit
  --outcsv OUTCSV, -o OUTCSV
                        Write resulting 4x4 transformation matrices
                        (flattened, row-major) to file.
  --verify              Whether to verify results by transforming reference
                        mesh and checking resulting verts.
  --debug
  --quiet
```