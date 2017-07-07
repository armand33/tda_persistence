# Persistence (TDA, INF563)

_Vincent Billaut, Armand Boschin, Théo Matussière_

## Aide à la lecture du rapport

Report : `report.pdf`

Filtrations in `filtrations/`

Code in `code/`


Exemple de code pour générer un barcode:

```bash
$ python3 test.py <filtration_filepath> <log> <save>
$ python3 test.py filtrations/torus.txt 0 barcodes/torus.png
```

    test.py filtration_source_file log [save]
    	filtration_source_file: file from which the filtration is extracted
    	log: int assessing that the x-scale be logarithmic
    	save: file in which to export the barcode (default=None)
