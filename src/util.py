import datasets
from datasets import Dataset

import pandas as pd


def load_radiology_dataset(file_path: str, set_name: str, sanity_check: bool = False, split: bool = False) -> Dataset:
    """Load the custom dataset from a CSV file and convert it to the necessary format.
    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    df = pd.read_csv(file_path)

    if sanity_check:
        df = df.sample(n=min(len(df), 1000), random_state=42)

    dataset = Dataset.from_pandas(df)

    dataset = dataset.rename_columns({"instruction": "prompt",
                                      "chosen_response": "chosen",
                                      "rejected_response": "rejected"})

    if set_name not in ['train', 'test']:
        raise ValueError(
            f"Split must be 'train' or 'test' but received: {set_name}")

    if split:
        dataset: datasets.DatasetDict = Dataset.train_test_split()
        return dataset[set_name]

    return dataset
