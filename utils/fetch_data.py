import requests
from tqdm import tqdm
from pathlib import Path

# links (from https://fever.ai/data.html)
links = {
    "Pre_processed_Wikipedia_Pages": "https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip",
    "Training_Dataset": "https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl",
    "Shared_Task_Development_Dataset_Labelled": "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl",
    "Shared_Task_Development_Dataset_Unlabelled": "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl",
    "Shared_Task_Blind_Test_Dataset_Unlabelled": "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl",
}


def download_file(url, store_folder="data"):
    fname = Path(url).name
    stored_path = Path(store_folder) / fname

    response = requests.get(url, stream=True, allow_redirects=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    print(f"[Downloading] {fname}")
    with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) as progress_bar:
        with open(stored_path, "wb") as dest_file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                dest_file.write(data)

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("*Error: something went wrong.")


if __name__ == "__main__":
    for name, url in links.items():
        fname = Path(url).name
        data_path = Path("data") / fname

        if data_path.is_file():
            print(f"*skip download file '{name}': already exists in {data_path}")
        else:
            download_file(url)
