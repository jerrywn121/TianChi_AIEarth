from sa_convlstm import SAConvLSTM
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from utils import *
import math


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(5)
        self.network = SAConvLSTM(configs.input_dim, configs.hidden_dim, configs.d_attn, configs.kernel_size).to(configs.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.3, patience=0, verbose=True, min_lr=0.0001)
        self.weight = torch.from_numpy(np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)).to(configs.device)

    def score(self, y_pred, y_true):
        with torch.no_grad():
            sc = score(y_pred, y_true, self.weight)
        return sc.item()

    def loss_sst(self, y_pred, y_true):
        # y_pred/y_true (N, 37, 24, 48)
        rmse = torch.mean((y_pred - y_true)**2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def loss_nino(self, y_pred, y_true):
        with torch.no_grad():
            rmse = torch.sqrt(torch.mean((y_pred - y_true)**2, dim=0)) * self.weight
        return rmse.sum()

    def train_once(self, sst, nino_true, ratio):
        sst_pred, nino_pred = self.network(sst.float()[:, :, None], teacher_forcing=True, 
                                           scheduled_sampling_ratio=ratio, train=True)
        self.optimizer.zero_grad()
        loss_sst = self.loss_sst(sst_pred, sst[:, 1:].to(self.device))
        loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
        loss_sst.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.optimizer.step()
        return loss_sst.item(), loss_nino.item(), nino_pred

    def test(self, dataloader_test):
        nino_pred = []
        sst_pred = []
        with torch.no_grad():
            for sst, _ in dataloader_test:
                sst, nino = self.network(sst.float()[:, :12, None], train=False)
                nino_pred.append(nino)
                sst_pred.append(sst)

        return torch.cat(sst_pred, dim=0), torch.cat(nino_pred, dim=0)

    def infer(self, dataset, dataloader):
        # calculate loss_func and score on a eval/test set
        self.network.eval()
        with torch.no_grad():
            sst_pred, nino_pred = self.test(dataloader)  # y_pred for the whole evalset
            nino_true = torch.from_numpy(dataset.target_nino).float().to(self.device)
            sst_true = torch.from_numpy(dataset.sst[:, 12:]).float().to(self.device)
            sc = self.score(nino_pred, nino_true)
            loss_sst = self.loss_sst(sst_pred, sst_true).item()
            loss_nino = self.loss_nino(nino_pred, nino_true).item()
        return loss_sst, loss_nino, sc, nino_pred

    def train(self, dataset_train, dataset_eval, chk_path):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = - math.inf
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            self.network.train()

            for j, (sst, nino_true) in enumerate(dataloader_train):
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                loss_sst, loss_nino, nino_pred = self.train_once(sst, nino_true, ssr_ratio)

                if j % self.configs.display_interval == 0:
                    sc = self.score(nino_pred, nino_true.float().to(self.device))
                    print('batch training loss: {:.2f}, {:.2f}, score: {:.4f}, ssr ratio: {:.4f}'.format(loss_sst, loss_nino, sc, ssr_ratio))

            # evaluation
            loss_sst_eval, loss_nino_eval, sc_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            print('epoch eval loss:\nsst: {:.2f}, nino: {:.2f}, sc: {:.4f}'.format(loss_sst_eval, loss_nino_eval, sc_eval))
            self.lr_scheduler.step(sc_eval)
            if sc_eval <= best:
                count += 1
                print('eval score is not improved for {} epoch'.format(count))
            else:
                count = 0
                print('eval score is improved from {:.5f} to {:.5f}, saving model'.format(best, sc_eval))
                self.save_model(chk_path)
                best = sc_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)


def prepare_data(ds_dir):
    # train/eval/test split
    cmip6sst, cmip5sst, cmip6nino, cmip5nino = read_raw_data(ds_dir)
    # if the processed data has been stored
    # cmip6sst, cmip5sst, cmip6nino, cmip5nino = read_from_nc(ds_dir)
    sst_train = [cmip6sst, cmip5sst[..., :-2]]
    nino_train = [cmip6nino, cmip5nino[..., :-2]]
    sst_eval = [cmip5sst[..., -2:-1]]
    nino_eval = [cmip5nino[..., -2:-1]]
    sst_test = [cmip5sst[..., -1:]]
    nino_test = [cmip5nino[..., -1:]]
    return sst_train, nino_train, sst_eval, nino_eval, sst_test, nino_test


if __name__ == '__main__':
    print(configs.__dict__)

    print('\nreading data')
    sst_train, nino_train, sst_eval, nino_eval, sst_test, nino_test = prepare_data('tcdata/enso_round1_train_20210201')

    print('processing training set')
    dataset_train = cmip_dataset(sst_train[0], nino_train[0], sst_train[1], nino_train[1], samples_gap=10)
    print(dataset_train.GetDataShape())
    del sst_train
    del nino_train
    print('processing eval set')
    dataset_eval = cmip_dataset(sst_cmip6=None, nino_cmip6=None,
                                sst_cmip5=sst_eval[0], nino_cmip5=nino_eval[0], samples_gap=5)
    print(dataset_eval.GetDataShape())
    del sst_eval
    del nino_eval
    trainer = Trainer(configs)
    trainer.save_configs('config_train.pkl')
    trainer.train(dataset_train, dataset_eval, 'checkpoint.chk')
    print('\n----- training finished -----\n')

    del dataset_train
    del dataset_eval

    print('processing test set')
    dataset_test = cmip_dataset(sst_cmip6=None, nino_cmip6=None,
                                sst_cmip5=sst_test[0], nino_cmip5=nino_test[0], samples_gap=1)
    print(dataset_test.GetDataShape())

    # test
    print('loading test dataloader')
    dataloader_test = DataLoader(dataset_test, batch_size=configs.batch_size_test, shuffle=False)
    chk = torch.load('checkpoint.chk')
    trainer.network.load_state_dict(chk['net'])
    print('testing...')
    loss_sst_test, loss_nino_test, sc_test = trainer.infer(dataset=dataset_test, dataloader=dataloader_test)
    print('test loss:\n sst: {:.2f}, nino: {:.2f}, score: {:.4f}'.format(loss_sst_test, loss_nino_test, sc_test))
