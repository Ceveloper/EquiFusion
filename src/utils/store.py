import os
import zarr
import numpy as np
import yaml


class SequenceStorer:
    def __init__(self, store_output_path, num_samples, num_sequences, config,if_gt=False):
        self.store_output_path = store_output_path
        self.num_samples = num_samples
        self.config = config
        self.if_gt = if_gt
        self.pred_length = self.config['pred_length']
        if config['if_eval_from60_at50fps']:
            self.pred_length = self.pred_length * 5 // 6
        os.makedirs(self.store_output_path, exist_ok=True)
        self.setup_store_preds()
        print(f"Storing output at {store_output_path}: ", num_sequences, 'segments')

    @property
    def future_shape(self):
        return (self.pred_length, self.config['num_joints'], 3)
    @property
    def past_shape(self):
        return (self.config['obs_length'], self.config['num_joints'], 3)

    def _setup_store_preds(self):
        store_output_path = os.path.join(self.store_output_path, 'output.zarr')
        chunk_size = int(10000 // self.num_samples)
        self.output_poses = zarr.open(store_output_path, mode='w', shape=(0, self.num_samples, *self.future_shape), 
                                    chunks=(chunk_size, self.num_samples, *self.future_shape), dtype=np.float32)

    def _setup_store_gt(self):
        self.gt_poses = zarr.open(os.path.join(self.store_output_path, 'gt.zarr'), mode='w', shape=(0, *self.future_shape), 
                                 chunks=(1000, *self.future_shape), dtype=np.float32)
        self.obs_poses = zarr.open(os.path.join(self.store_output_path, 'obs.zarr'), mode='w', shape=(0, *self.past_shape), 
                            chunks=(1000, *self.past_shape), dtype=np.float32)

    def setup_store_preds(self):
        if self.if_gt:
            self._setup_store_gt()
        else:
            self._setup_store_preds()
        self.output_metadata = {'unique_id': [], 'metadata':[]}

    def store_batch(self, output, extra, dataset):
        if self.if_gt:
            gt, obs = output
            self.gt_poses.append(dataset.skeleton.if_add_zero_pad_center_hip(gt).cpu().numpy(), axis=0)
            self.obs_poses.append(dataset.skeleton.if_add_zero_pad_center_hip(obs).cpu().numpy(), axis=0)
        else:
            self.output_poses.append(dataset.skeleton.if_add_zero_pad_center_hip(output).cpu().numpy(), axis=0)
        self.output_metadata['unique_id'].extend(dataset.unique_sample_string(extra))
        self.output_metadata['metadata'].extend(extra['metadata'][dataset.metadata_class_idx])

    def finalize_store(self):
        with open(os.path.join(self.store_output_path, 'metadata.yaml'), 'w') as outfile:
            yaml.dump(self.output_metadata, outfile, default_flow_style=False)