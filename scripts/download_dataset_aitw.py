from pathlib import Path
from google.cloud import storage

client = storage.Client.create_anonymous_client()  # public bucket
bucket = client.bucket("gresearch")

dataset_path  = Path("data/raw/")

blobs = bucket.list_blobs(prefix="android-in-the-wild/single")
for blob in blobs:
    if blob.name.endswith("/"):  # skip GCS directory-marker objects
        continue
    local_path = dataset_path / blob.name
    print(f"local path={local_path}")
    local_path.parent.mkdir(exist_ok=True, parents=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded: {blob.name}")