import numpy as np
import os

import pandas as pd
from .base import MotionDataset

amass_official_splits = {
        'validation': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']#
    }

class AMASSDataset(MotionDataset):
    def __init__(self, datasets,  *args, file_idces='all', **kwargs):


        self.datasets, self.file_idces = datasets, file_idces
        assert self.file_idces == "all", "We only support 'all' files for now"

        self.FPS = 60
        
        self.dict_indices = {} # dict_indices[dataset][file_idx] indicated idx where dataset-file_idx annotations start.
        self.mm_indces = None
        self.metadata_class_idx = 0 # 0: dataset, 1: filename --> dataset is the class used for metrics computation
        self.idx_to_class = ['DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA', 'SSM', 'Transitions']
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}

        super().__init__(datasets=datasets, **kwargs)
        assert (datasets is not None and file_idces is not None) or self.segments_path is not None
            

    def _prepare_data(self, num_workers=8):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path, num_workers=num_workers)
            self.stride = 1
            self.augmentation = 0
        else:
            self.annotations = self._read_all_annotations(self.datasets, self.file_idces)
            self.segments, self.segment_idx_to_metadata = self._generate_segments()

    def _read_all_annotations(self, datasets, file_idces):
        preprocessed_path = os.path.join(self.precomputed_folder, f'data_3d_amass.npz')
        if not os.path.exists(self.precomputed_folder):
            raise NotImplementedError("Preprocessing of AMASS dataset is not implemented yet. Please use the preprocessed data.")

        # we load from already preprocessed dataset
        data_o = np.load(preprocessed_path, allow_pickle=True)['positions_3d'].item()
        
        anns_all = []
        self.dict_indices = {}
        self.clip_idx_to_metadata = []
        counter = 0
        n_frames = 0
        
        print("Loading datasets: ", datasets, file_idces)
        for dataset in datasets:
            self.dict_indices[dataset] = {}
            
            # we build the feature vectors for each dataset and file_idx
            for file_idx in list(data_o[dataset].keys()):

                seq =data_o[dataset][file_idx]

                self.dict_indices[dataset][file_idx] = counter
                self.clip_idx_to_metadata.append((dataset, str(file_idx)))
                counter += 1
                n_frames += len(seq)

                anns_all.append(seq.astype(self.dtype)) # datasets axis expanded

        centered_anns = [ann - ann[0:1, 0:1, :] for ann in anns_all]
        centered_cat_anns = np.concatenate(centered_anns , axis=0)
        centered_cat_anns_no_trans = centered_cat_anns - centered_cat_anns[:, 0:1, :]
        # test 1.095772 -1.0687416
        # train  1.1635599 -1.12504
        # valid 1.1759695 -1.1325874
        return anns_all

    def _load_annotations_and_segments(self, segments_path, num_workers=8):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        # columns -> dataset,file,file_idx,pred_init,pred_end
        datasets, file_idces = list(df["dataset"].unique()), list(df["file_idx"].unique())
        self.annotations = self._read_all_annotations(datasets, "all")#file_idces)
        
        segments = [(self.dict_indices[row["dataset"]][row["file_idx"]], 
                    row["pred_init"] - self.obs_length, 
                    row["pred_init"] + self.pred_length - 1) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(row["dataset"], str(row["file_idx"])) for i, row in df.iterrows()]
                        
        return segments, segment_idx_to_metadata