from longvideobench import LongVideoBenchDataset
YOUR_DATA_PATH = "/mnt/Shared_03/fsq/LongVideoBench"

# validation
dataset = LongVideoBenchDataset(YOUR_DATA_PATH, "lvb_val.json", max_num_frames=64)

# test
# dataset = LongVideoBenchDataset(YOUR_DATA_PATH, "lvb_test_wo_gt.json", max_num_frames=64)

print(dataset[0]["inputs"]) # A list consisting of PIL.Image and strings.


import pdb
pdb.set_trace()