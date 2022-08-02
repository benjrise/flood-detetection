import csv
from config import FoundationConfig
from dataclasses import asdict

config = FoundationConfig()
out = asdict(config)
w = csv.writer(open("output.csv", "w"))
# loop over dictionary keys and values
for key, val in out.items():
    # write every key and value to file
    w.writerow([key, val])
