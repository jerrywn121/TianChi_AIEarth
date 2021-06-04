from transformer import SpaceTimeTransformer
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
from utils import *
import math


class NoamOpt:
    """
    learning rate warmup and decay
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(5)
        self.network = SpaceTimeTransformer(configs).to(configs.device)
        adam = torch.optim.Adam(self.network.parameters(), lr=0, weight_decay=configs.weight_decay)
        factor = math.sqrt(configs.d_model*configs.warmup)*0.0014
        self.opt = NoamOpt(configs.d_model, factor, warmup=configs.warmup, optimizer=adam)
        self.weight = torch.from_numpy(np.array([1.5]*4 + [2]*7 + [3]*7 + [4]*6) * np.log(np.arange(24)+1)).to(configs.device)

    def score(self, y_pred, y_true):
        # compute for nino_pred, nino_true
        with torch.no_grad():
            sc = score(y_pred, y_true, self.weight)
        return sc.item()

    def loss_sst(self, y_pred, y_true):
        # y_pred/y_true (N, 26, 24, 48)
        rmse = torch.mean((y_pred - y_true)**2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def loss_nino(self, y_pred, y_true):
        with torch.no_grad():
            rmse = torch.sqrt(torch.mean((y_pred - y_true)**2, dim=0)) * self.weight
        return rmse.sum()

    def train_once(self, input_sst, sst_true, nino_true, ssr_ratio):
        sst_pred, nino_pred = self.network(src=input_sst.float().to(self.device),
                                           tgt=sst_true[:, :, None].float().to(self.device),
                                           train=True, ssr_ratio=ssr_ratio)
        self.opt.optimizer.zero_grad()
        loss_sst = self.loss_sst(sst_pred, sst_true.float().to(self.device))
        loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
        loss_sst.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.opt.step()
        return loss_sst.item(), loss_nino.item(), nino_pred

    def test(self, dataloader_test):
        nino_pred = []
        sst_pred = []
        with torch.no_grad():
            for input_sst, sst_true, nino_true in dataloader_test:
                sst, nino = self.network(src=input_sst.float().to(self.device),
                                         tgt=None, train=False)
                nino_pred.append(nino)
                sst_pred.append(sst)

        return torch.cat(sst_pred, dim=0), torch.cat(nino_pred, dim=0)

    def infer(self, dataset, dataloader):
        self.network.eval()
        with torch.no_grad():
            sst_pred, nino_pred = self.test(dataloader)
            nino_true = torch.from_numpy(dataset.target_nino).float().to(self.device)
            sst_true = torch.from_numpy(dataset.target_sst).float().to(self.device)
            sc = self.score(nino_pred, nino_true)
            loss_sst = self.loss_sst(sst_pred, sst_true).item()
            loss_nino = self.loss_nino(nino_pred, nino_true).item()
        return loss_sst, loss_nino, sc

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
            # train
            self.network.train()
            for j, (input_sst, sst_true, nino_true) in enumerate(dataloader_train):
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                loss_sst, loss_nino, nino_pred = self.train_once(input_sst, sst_true, nino_true, ssr_ratio)  # y_pred for one batch

                if j % self.configs.display_interval == 0:
                    sc = self.score(nino_pred, nino_true.float().to(self.device))
                    print('batch training loss: {:.2f}, {:.2f}, score: {:.4f}, ssr: {:.5f}, lr: {:.5f}'.format(loss_sst, loss_nino, sc, ssr_ratio, self.opt.rate()))

                # increase the number of evaluations in order not to miss the optimal point 
                # which is feasible because of the less training time of timesformer
                if (i+1 >= 9) and (j+1)%300 == 0:
                    _, _, sc_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
                    print('epoch eval loss: sc: {:.4f}'.format(sc_eval))
                    if sc_eval > best:
                        self.save_model(chk_path)
                        best = sc_eval
                        count = 0

            # evaluation
            loss_sst_eval, loss_nino_eval, sc_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            print('epoch eval loss:\nsst: {:.2f}, nino: {:.2f}, sc: {:.4f}'.format(loss_sst_eval, loss_nino_eval, sc_eval))
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
                    'optimizer': self.opt.optimizer.state_dict()}, path)


def prepare_data(ds_dir):
    # train/eval/test split
    cmip6sst, cmip5sst, cmip6nino, cmip5nino = read_raw_data(ds_dir)
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
    dataset_train = cmip_dataset(sst_train[0], nino_train[0], sst_train[1], nino_train[1], samples_gap=5)
    print(dataset_train.GetDataShape())
    del sst_train
    del nino_train
    print('processing eval set')
    dataset_eval = cmip_dataset(sst_cmip6=None, nino_cmip6=None,
                                sst_cmip5=sst_eval[0], nino_cmip5=nino_eval[0], samples_gap=1)
    print(dataset_eval.GetDataShape())
    del sst_eval
    del nino_eval
    trainer = Trainer(configs)
    trainer.save_configs('config_train.pkl')
    trainer.train(dataset_train, dataset_eval, 'checkpoint.chk')
