# Data Directory Overview

This directory contains various datasets and associated Jupyter notebook projects used in our study, organized into four main subdirectories. Each subdirectory serves a specific purpose, as detailed below:

## Subdirectories

### `ori_data`

This folder contains the original, raw data collected for our study. These datasets are unprocessed and serve as the basis for all further analyses and processing steps. Accompanying these data files are Jupyter notebook files that detail the data processing steps used to generate datasets for subsequent phases of the project. 

**Contents**:
- Raw data files: Include various formats depending on the source and nature of the collected information.
- Data processing notebooks: Jupyter notebooks containing the code used to process the raw data into formatted datasets suitable for analysis, testing, and training.

### `test_task_data`

The `test_task_data` subdirectory holds data derived from the original datasets (`ori_data`) that has been processed and structured for testing task settings. This includes data specifically formatted and organized to simulate real-world scenarios or conditions under which our models or algorithms are expected to operate. This data is used exclusively for testing purposes.

**Intended use**:
- Testing scenarios: Data organized to facilitate simulation of specific scenarios or tasks for model validation.

### `internal_classifier_data`

In the `internal_classifier_data` folder, you'll find data that has been processed from the original datasets and is used to train internal classifiers. These classifiers are integral to our method and are employed to score or evaluate certain attributes within our system. 

**Intended use**:
- Internal model training: Structured data used for training our proprietary classification algorithms.

### `external_classifier_data`

The `external_classifier_data` directory contains processed data from the original datasets intended for training external classifiers. These classifiers are used to assess the effectiveness of our continuation methods or other external benchmarks. This data is crucial for evaluating how well our approaches perform compared to standard or existing methodologies.

**Intended use**:
- External model evaluation: Data sets designed for benchmarking and comparative analysis with external systems or models.

## General Guidelines

- **Do not modify or delete** data in the `ori_data` directory to preserve the integrity of the original datasets.
- Ensure that any processed data placed in the subdirectories maintains consistent formatting and structure for compatibility with our testing and training frameworks.
- Refer to the Jupyter notebooks in the `ori_data` folder for details on the data processing steps and how to generate datasets for specific uses.

## Additional Information

For more detailed information about the data processing steps, methodologies, or specific file formats, please refer to the project's main documentation or contact the data team.

Thank you for adhering to the data organization and usage protocols.
