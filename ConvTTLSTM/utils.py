import numpy as np
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
import torch


def prepare_inputs_targets(len_time, input_gap, input_length, pred_shift, pred_length, samples_gap):
    # input_gap=1: time gaps between two consecutive input frames
    # input_length=12: the number of input frames
    # pred_shift=26: the lead_time of the last target to be predicted
    # pred_length=26: the number of frames to be predicted
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    pred_gap = pred_shift // pred_length
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span+pred_shift-1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones((1, input_length+pred_length), dtype=int)
    return ind[::samples_gap]


def fold(data, size=36, stride=12):
    # inverse of unfold/sliding window operation
    # only applicable to the case where the size of the sliding windows is n*stride
    # data (N, size, *)
    # outdata (N_, *)
    # N/size is the number/width of sliding blocks
    assert size % stride == 0
    times = size // stride
    remain = (data.shape[0] - 1) % times
    if remain > 0:
        ls = list(data[::times]) + [data[-1, -(remain*stride):]]
        outdata = np.concatenate(ls, axis=0)  # (36*(151//3+1)+remain*stride, *, 15)
    else:
        outdata = np.concatenate(data[::times], axis=0)  # (36*(151/3+1), *, 15)
    assert outdata.shape[0] == size * ((data.shape[0]-1)//times+1) + remain * stride
    return outdata


def data_transform(data, num_years_per_model):
    # data (N, 36, *)
    # num_years_per_model: 151/140
    length = data.shape[0]
    assert length % num_years_per_model == 0
    num_models = length // num_years_per_model
    outdata = np.stack(np.split(data, length/num_years_per_model, axis=0), axis=-1)  # (151, 36, *, 15)
    outdata = fold(outdata)
    # check output data
    assert outdata.shape[-1] == num_models
    assert not np.any(np.isnan(outdata))
    return outdata


def read_raw_data(ds_dir, out_dir=None):
    # read and process raw cmip data from CMIP_train.nc and CMIP_label.nc
    train_cmip = xr.open_dataset(Path(ds_dir) / 'CMIP_train.nc').transpose('year', 'month', 'lat', 'lon')
    label_cmip = xr.open_dataset(Path(ds_dir) / 'CMIP_label.nc').transpose('year', 'month')

    # select longitudes
    lon = train_cmip.lon.values
    lon = lon[np.logical_and(lon>=95, lon<=330)]
    train_cmip = train_cmip.sel(lon=lon)

    cmip6sst = data_transform(train_cmip.sst.values[:2265], 151) 
    cmip5sst = data_transform(train_cmip.sst.values[2265:], 140)
    cmip6nino = data_transform(label_cmip.nino.values[:2265], 151) 
    cmip5nino = data_transform(label_cmip.nino.values[2265:], 140)

    assert len(cmip6sst.shape) == 4
    assert len(cmip5sst.shape) == 4
    assert len(cmip6nino.shape) == 2
    assert len(cmip5nino.shape) == 2

    # store processed data for faster data access
    if out_dir is not None:
        ds_cmip6 = xr.Dataset({'sst': (['month', 'lat', 'lon', 'model'], cmip6sst),
                               'nino': (['month', 'model'], cmip6nino)},
                              coords={'month': np.repeat(np.arange(1, 13)[None], cmip6nino.shape[0] // 12, axis=0).flatten(),
                                      'lat': train_cmip.lat.values, 'lon': train_cmip.lon.values,
                                      'model': np.arange(15)+1})
        ds_cmip6.to_netcdf(Path(out_dir) / 'cmip6.nc')
        ds_cmip5 = xr.Dataset({'sst': (['month', 'lat', 'lon', 'model'], cmip5sst),
                               'nino': (['month', 'model'], cmip5nino)},
                              coords={'month': np.repeat(np.arange(1, 13)[None], cmip5nino.shape[0] // 12, axis=0).flatten(),
                                      'lat': train_cmip.lat.values, 'lon': train_cmip.lon.values,
                                      'model': np.arange(17)+1})
        ds_cmip5.to_netcdf(Path(out_dir) / 'cmip5.nc')
    train_cmip.close()
    label_cmip.close()
    return cmip6sst, cmip5sst, cmip6nino, cmip5nino


def read_from_nc(ds_dir):
    # an alternative for reading processed data
    cmip6 = xr.open_dataset(Path(ds_dir) / 'cmip6.nc').transpose('month', 'lat', 'lon', 'model')
    cmip5 = xr.open_dataset(Path(ds_dir) / 'cmip5.nc').transpose('month', 'lat', 'lon', 'model')
    return cmip6.sst.values, cmip5.sst.values, cmip6.nino.values, cmip5.nino.values



def score(y_pred, y_true, acc_weight):
    # for pytorch
    # acc_weight = np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)
    # acc_weight = torch.from_numpy(acc_weight).to(device)
    pred = y_pred - y_pred.mean(dim=0, keepdim=True)  # (N, 24)
    true = y_true - y_true.mean(dim=0, keepdim=True)  # (N, 24)
    cor = (pred * true).sum(dim=0) / (torch.sqrt(torch.sum(pred**2, dim=0) * torch.sum(true**2, dim=0)) + 1e-6)
    acc = (acc_weight * cor).sum()
    rmse = torch.mean((y_pred - y_true)**2, dim=0).sqrt().sum()
    return 2/3. * acc - rmse


def cat_over_last_dim(data):
    return np.concatenate(np.moveaxis(data, -1, 0), axis=0)


class cmip_dataset(Dataset):
    def __init__(self, sst_cmip6, nino_cmip6, sst_cmip5, nino_cmip5, samples_gap):
        super().__init__()
        # cmip6 (N, *, 15)
        # cmip5 (N, *, 17)
        sst = []
        target_nino = []
        if sst_cmip6 is not None:
            assert len(sst_cmip6.shape) == 4
            assert len(nino_cmip6.shape) == 2
            idx_sst = prepare_inputs_targets(sst_cmip6.shape[0], input_gap=1, input_length=12, 
                                             pred_shift=26, pred_length=26, samples_gap=samples_gap)

            sst.append(cat_over_last_dim(sst_cmip6[idx_sst]))
            target_nino.append(cat_over_last_dim(nino_cmip6[idx_sst[:, 12:36]]))
        if sst_cmip5 is not None:
            assert len(sst_cmip5.shape) == 4
            assert len(nino_cmip5.shape) == 2
            idx_sst = prepare_inputs_targets(sst_cmip5.shape[0], input_gap=1, input_length=12, 
                                             pred_shift=26, pred_length=26, samples_gap=samples_gap)
            sst.append(cat_over_last_dim(sst_cmip5[idx_sst]))
            target_nino.append(cat_over_last_dim(nino_cmip5[idx_sst[:, 12:36]]))

        # sst data containing both the input and target
        self.sst = np.concatenate(sst, axis=0)  # (N, 38, lat, lon)
        # nino data containing the target only
        self.target_nino = np.concatenate(target_nino, axis=0)  # (N, 24)
        assert self.sst.shape[0] == self.target_nino.shape[0]
        assert self.sst.shape[1] == 38
        assert self.target_nino.shape[1] == 24

    def GetDataShape(self):
        return {'sst': self.sst.shape,
                'nino target': self.target_nino.shape}

    def __len__(self):
        return self.sst.shape[0]

    def __getitem__(self, idx):
        return self.sst[idx], self.target_nino[idx]
