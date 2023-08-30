# Dataset Preparation

## Data Source
We use [The Material Project](https://next-gen.materialsproject.org/) as the source of the dataset generation.

## Usage

### Prepare API_KEY for The Material Project

1. Create an account: https://profile.materialsproject.org/
2. Visit and generate your API_KEY: https://next-gen.materialsproject.org/api#api-key
3. Save your API_KEY as "API_KEY.txt" in this directory

### Running Scripts

- The following commands assume you are running in `/workspace` in the Docker environment.
- The working directory is `/workspace/data` and finally `/workspace/data/dataset/*` will be created.

1. Download The Material Project data.
    - `poetry run python src/dataset_generation/01_download.py`.
2. Extract structures from the crawled data.
    - `poetry run python src/dataset_generation/02_extract_structures.py`.
3. Convert to a conventional cell.
    -`poetry run python src/dataset_generation/03_convert_to_conventional_cell.py`.
4. Select materials according to each criterion.
    - `poetry run python src/dataset_generation/04_01_select_materials_by_cell.py`.
      - For the lim_l6 dataset.
    - `poetry run python src/dataset_generation/04_02_select_materials_by_formula.py`.
        - For the ICSG3D dataset.
    - Run python src/dataset_generation/04_03_select_materials_by_list.py.
        - For the YBCO13 dataset.
7. Split the dataset into train/validation/test splits.
    - `poetry run python src/dataset_generation/05_split.py`.
8. Generate the NeSF style dataset.
    - `poetry run python src/dataset_generation/06_generate_NeSF_dataset.py`.