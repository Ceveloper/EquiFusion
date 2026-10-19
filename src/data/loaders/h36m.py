import numpy as np
import os
import pandas as pd
import os

from .base import MotionDataset

class H36MDataset(MotionDataset):
    def __init__(self,  subjects,  *args, actions='all', **kwargs): # data augmentation strategies


        self.subjects, self.actions = subjects, actions
        self.FPS = 50

        self.dict_indices = {} # dict_indices[subject][action] indicated idx where subject-action annotations start.
        self.mm_indces = None
        self.metadata_class_idx = 1 # 0: subject, 1: action --> action is the class used for metrics computation
        self.idx_to_class = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}

        super().__init__(*args, actions=actions, **kwargs)
        assert (subjects is not None and actions is not None) or self.segments_path is not None
   
    def load_mmgt(self):
        if self.split == 'train':
            if 'S8' in self.subjects:
                mmgt_path = os.path.join(self.precomputed_folder, f"mmgt_{self.split}.txt")
            else: 
                mmgt_path = os.path.join(self.precomputed_folder, f"mmgt_{self.split}_noS8.txt")
            super().load_mmgt(mmgt_path)
        else:
            super().load_mmgt()
           

    def _prepare_data(self, num_workers=8):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path, num_workers=num_workers)
            self.stride = 1
            self.augmentation = 0
        else:
            self.annotations = self._read_all_annotations(self.subjects, self.actions)
            self.segments, self.segment_idx_to_metadata = self._generate_segments()
            
    def _read_all_annotations(self, subjects, actions):
        preprocessed_path = os.path.join(self.precomputed_folder, f'data_3d_h36m.npz')

        # we load from already preprocessed dataset
        data_o = np.load(preprocessed_path, allow_pickle=True)['positions_3d'].item()
        data_f = dict(filter(lambda x: x[0] in subjects, data_o.items()))
        if actions != 'all': # if not all, we only keep the data from the selected actions, for each participant
            for subject in list(data_f.keys()):
                data_f[subject] = dict(filter(lambda x: any([a in x[0] for a in actions]), data_f[subject].items()))
                if len(data_f[subject]) == 0: # no actions for subject => delete
                    data_f.pop(subject)
                    print(f"Subject '{subject}' has no actions available from '{actions}'.")
                for action in data_f[subject].keys():
                    assert not np.isnan(data_f[subject][action]).any()
        else:
            if not self.silent:
                print(f"All actions loaded from {subjects}.")

        self.data = data_f

        anns_all = []
        self.dict_indices = {}
        self.clip_idx_to_metadata = []
        counter = 0
        for subject in self.data:
            self.dict_indices[subject] = {}

            for action in self.data[subject]:
                self.dict_indices[subject][action] = counter
                self.clip_idx_to_metadata.append((subject, action.split(" ")[0].split("_")[0]))
                counter += 1

                anns_all.append(self.data[subject][action].astype(self.dtype)) # participants axis expanded
        centered_anns = [ann - ann[0:1, 0:1, :] for ann in anns_all]
        centered_cat_anns = np.concatenate(centered_anns , axis=0)
        centered_cat_anns_no_trans = centered_cat_anns - centered_cat_anns[:, 0:1, :]
        # train  0.98306274 -0.9905092
        # valid  0.9272223 -0.91470855
        
        self._generate_statistics_full(anns_all)

        return anns_all

    def _load_annotations_and_segments(self, segments_path, num_workers=8):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        df["action"] = df["action"].apply(lambda x: x.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog').replace(' ', '_'))
        subjects, actions = list(df["subject"].unique()), list(df["action"].unique())
        if not self.silent:
            print(subjects, actions)
        self.annotations = self._read_all_annotations(subjects, actions)
        segments = [(self.dict_indices[row["subject"]][row["action"]], 
                    int(row["init"]), 
                    int(row["pred_end"])) 
                        for i, row in df.iterrows()]
        
        segment_idx_to_metadata = [(row["subject"], row["action"].split(" ")[0].split("_")[0]) for i, row in df.iterrows()]
                        
        return segments, segment_idx_to_metadata
