import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
from torch.utils.data import Dataset, Subset
import os


class MD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    # raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    # molecule_files = dict(
    #     aspirin="aspirin_dft.npz",
    #     benzene="benzene2017_dft.npz",
    #     ethanol="ethanol_dft.npz",
    #     malonaldehyde="malonaldehyde_dft.npz",
    #     naphthalene="naphthalene_dft.npz",
    #     salicylic_acid="salicylic_dft.npz",
    #     toluene="toluene_dft.npz",
    #     uracil="uracil_dft.npz",
    # )

    # # We note that the file names have been changed. # ok I can see 
    # # For example, `aspirin_dft` -> `md17_aspirin`
    # # See https://github.com/pyg-team/pytorch_geometric/commit/213f0ff95140eb1a1fbf7d99b012d458ef360f71#diff-a85570faabaf1806684e5b6654deed3863273bbe703f237846accd11948f4675
    # molecule_files = dict(
    #     aspirin="md17_aspirin.npz",
    #     benzene="md17_benzene2017.npz",
    #     ethanol="md17_ethanol.npz",
    #     malonaldehyde="md17_malonaldehyde.npz",
    #     naphthalene="md17_naphthalene.npz",
    #     salicylic_acid="md17_salicylic.npz",
    #     toluene="md17_toluene.npz",
    #     uracil="md17_uracil.npz",
    # )

    # available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None):
        """
        example:
            root: "/home/sire/phd/srz228573/equiformer/data_sl/equiformer_data/md17/aspirin"
            dataset_arg: aspirin

        """
        # assert dataset_arg is not None, (
        #     "Please provide the desired comma separated molecule(s) through"
        #     f"'dataset_arg'. Available molecules are {', '.join(MD17.available_molecules)} "
        #     "or 'all' to train on the combined dataset."
        # )
        # assert dataset_arg in MD17.available_molecules, "Unknown data argument"

        # For simplicity, always use one type of molecules 
        '''
        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.available_molecules)
        '''
        #self.molecules = dataset_arg.split(",")# ['aspirin']

        '''
        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )
        '''

        super(MD17, self).__init__(root, transform, pre_transform)# it runs the process function

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:# ['/home/sire/phd/srz228573/equiformer/data_sl/equiformer_data/md17/aspirin/processed/md17-aspirin.pt']
            data, slices = torch.load(path)#
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )
            # Extract the directory path
            directory_path = os.path.dirname(path)
            import shutil
            shutil.rmtree(directory_path) # remove processed folder as it is not needed anymore
        

    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(MD17, self).get(idx - self.offsets[data_idx])

    @property
    def raw_file_names(self):
        return ["nequip_npz.npz"]#[MD17.molecule_files[mol] for mol in self.molecules]# this name of npz file to be loaded

    @property
    def processed_file_names(self):
        return ["md17.pt"]# ['/home/sire/phd/srz228573/equiformer/data_sl/equiformer_data/md17/aspirin/processed/md17-aspirin.pt']

    # def download(self):
    #     for file_name in self.raw_file_names:
    #         download_url(MD17.raw_url + file_name, self.raw_dir)

    def process(self):
        path = self.root + "/nequip_npz.npz"
        # for path in self.raw_paths:# output of function raw_file_names
        data_npz = np.load(path)#/home/sire/phd/srz228573/equiformer/data_sl/equiformer_data/md17/aspirin/raw/md17_aspirin.npz
        z = torch.from_numpy(data_npz["atomic_number"]).long()
        positions = torch.from_numpy(data_npz['pos']).float()
        energies = torch.from_numpy(data_npz['energy']).float()
        forces = torch.from_numpy(data_npz["force"]).float()
        
        samples = []
        for pos, y, dy in zip(positions, energies, forces):
            #import pdb as pdb
            #pdb.set_trace()
            y = torch.unsqueeze(y, 0)
            samples.append(Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy))

        if self.pre_filter is not None:
            samples = [data for data in samples if self.pre_filter(data)]

        if self.pre_transform is not None:
            samples = [self.pre_transform(data) for data in samples]

        data, slices = self.collate(samples)
        torch.save((data, slices), self.root + "/processed/md17.pt")# ceates processed files


def get_md17_datasets(root):
    '''
        Return training, validation and testing sets of MD17 with the same data partition as TorchMD-NET.
    '''

    all_dataset = MD17(root)
    
    # hack for using subset class as in original
    subset_indices = list(range(len(all_dataset)))
    subset = Subset(all_dataset, subset_indices)
    train_dataset = subset

    return train_dataset


if __name__ == '__main__':

    from torch_geometric.loader import DataLoader

    _root_path = './test_md17/aspirin'
    train_dataset, val_dataset, test_dataset = get_md17_datasets(root=_root_path, 
        dataset_arg='aspirin', 
        train_size=950, val_size=50, test_size=None, 
        seed=1)

    print('Training set size:   {}'.format(len(train_dataset)))
    print('Validation set size: {}'.format(len(val_dataset)))
    print('Testing set size:    {}'.format(len(test_dataset)))

    print(train_dataset[2])

    train_loader = DataLoader(train_dataset, batch_size=8)
    for i, data in enumerate(train_loader):
        print(data)
        print(data.y)
        break
        