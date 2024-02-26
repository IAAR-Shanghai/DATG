# FUDGE Method Implementation

This repository contains reproduction of FUDGE method as proposed in the paper *FUDGE: Controlled Text Generation With Future Discriminators*, available [here](https://aclanthology.org/2021.naacl-main.276/). This baseline method utilizes future discriminators for controlled text generation, offering an innovative approach to influencing the generation process of language models.

## File Descriptions

- `fudge.py`: This is the main file implementing the FUDGE method. 

- `classifier.py`: This file contains the architecture of the classifiers used by the FUDGE method.

- `logits_processor.py`: Stores the class responsible for adjusting the decoding logits using the classifiers as per the FUDGE methodology.
