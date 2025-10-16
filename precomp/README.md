Prerequisite:

In ./resources have jstor_metadata_2025-10-09.jsonl.gz
./resources/jstor_metadata_2025-10-09.jsonl.gz

Or the most up to date jtor metadata found here:
https://support.stewardship.jstor.org/hc/en-us/articles/15581546658967-Metadata-fields-Tier-1

Then enter a venv and pip install -r requirements.txt 

Then run: python precomp.py

Which will embed the jsonl into a primray and secondary segmented faiss files, then quantize the secondary segmented faiss files, then organize the files into proper locations (deleting unimportant files)