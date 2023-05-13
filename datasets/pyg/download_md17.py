import md17 as md17_dataset

def main():
    for i in ["aspirin", "benzene", "ethanol", "malonaldehyde", "naphthalene", "salicylic_acid", "toluene", "uracil"]:
        print(i)
        md17_dataset.get_md17_datasets(root="/home/sire/phd/srz228573/scratch/benchmarking_datasets/equiformer_data/md17", dataset_arg = i, train_size = 950, val_size = 50, test_size = None, seed = 1)


if __name__ == "__main__":
    main()
    

