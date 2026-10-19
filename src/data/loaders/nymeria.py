import numpy as np
import os
import pandas as pd

from .base import MotionDataset


class NymeriaDataset(MotionDataset):
    dataset_name = 'nymeria'
    def __init__(self, *args, annotations_folder=None, **kwargs): 
                
        self.FPS = 60
        self.annotations_folder = annotations_folder
        
        self.dict_indices = {} # dict_indices[seq_name] indicated idx where subject-action annotations start.
        self.mm_indces = None
        self.metadata_class_idx = 0 # 0: se_name, 1: fake_labelfor allseqs -->the idx sets the value for metrics computation

        super().__init__(*args, **kwargs)
        assert self.split != 'test' or self.segments_path is not None

    def extract_action_label(self, extra):
        return extra['metadata'][0]    

    
    def _prepare_data(self, num_workers=8):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path, num_workers=num_workers)
            self.stride = 1
            self.augmentation = 0
        else:
            self.annotations = self._read_all_annotations(self.split)
            self.segments, self.segment_idx_to_metadata = self._generate_segments()
            
    def _read_all_annotations(self, split):
        preprocessed_path = os.path.join(self.precomputed_folder, f'data_3d_{self.dataset_name}.npz')
        
        data_o = np.load(preprocessed_path, allow_pickle=True)['positions_3d'].item()
        data_f = data_o[split]

        self.idx_to_class = [seq for seq in list(data_f.keys())]
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}

        anns_all = []
        self.dict_indices = {}
        self.clip_idx_to_metadata = []
        counter = 0
        for seq_name in data_f:
            self.dict_indices[seq_name] = counter
            self.clip_idx_to_metadata.append((seq_name, seq_name))
            counter += 1
            data_f[seq_name] = data_f[seq_name]
            anns_all.append(data_f[seq_name].astype(self.dtype))
        
        self._generate_statistics_full(anns_all)
        centered_anns = [ann - ann[0:1, 0:1, :] for ann in anns_all]
        centered_cat_anns = np.concatenate(centered_anns , axis=0)
        centered_cat_anns_no_trans = centered_cat_anns - centered_cat_anns[:, 0:1, :]
        print("min and max values of centered sequences: ", centered_cat_anns_no_trans .max(), centered_cat_anns_no_trans .min())
        
        return anns_all
    
    def _load_annotations_and_segments(self, segments_path, num_workers=8):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        self.annotations = self._read_all_annotations(self.split)
        segments = [(self.dict_indices[row["name"]], 
                    int(row["init"]),
                    int(row["pred_end"])) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(row["name"], row["name"]) for i, row in df.iterrows()]
                        
        return segments, segment_idx_to_metadata
           
