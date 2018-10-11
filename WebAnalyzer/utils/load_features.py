import os
import h5py
import torch
import numpy as np

def load_features():
    features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', '..', 'data', 'features')

    features=dict()
    for dataset in os.listdir(features_path):
        descs=dict()
        for desc in os.listdir(os.path.join(features_path,dataset)):
            feat = []
            name = []
            files=os.listdir(os.path.join(features_path,dataset,desc))
            for f in files:
                feature_file=h5py.File(os.path.join(os.path.join(features_path,dataset,desc,f)))
                feat.append(feature_file[desc][()])
                name.append(feature_file['names'][()])
                feature_file.close()
            feat_np=np.concatenate(feat,axis=0)
            name_np=np.concatenate(name,axis=0)

            descs[desc]=torch.tensor(feat_np.reshape([feat_np.shape[0],-1])).cuda()
            descs['names']=name_np
            print(descs['names'].shape)

        features[dataset]=descs

    print(features.keys())
    print(features['photo'].keys())
    return features

