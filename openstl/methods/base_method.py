from typing import Dict, List, Union
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from contextlib import suppress
from timm.utils import NativeScaler
from timm.utils.agc import adaptive_clip_grad

from openstl.core import metric
from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.utils import gather_tensors_batch, get_dist_info, ProgressBar
from openstl.utils.main_utils import mis_loss_func, eval_quantiles
from openstl.core.metrics import MAE, MSE

has_native_amp = False
import pdb
try:

    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

from torch.autograd import grad
class Base_method(object):
    """Base Method.

    This class defines the basic functions of a video prediction (VP)
    method training and testing. Any VP method that inherits this class
    should at least define its own `train_one_epoch`, `vali_one_epoch`,
    and `test_one_epoch` function.

    """

    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.dist = args.dist
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        if self.dist:
            self.rank, self.world_size = get_dist_info()
            assert self.rank == int(device.split(':')[-1])
        else:
            self.rank, self.world_size = 0, 1
        self.clip_value = self.args.clip_grad
        self.clip_mode = self.args.clip_mode if self.clip_value is not None else None
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        # setup metrics
        if 'weather' in self.args.dataname:
            self.metric_list, self.spatial_norm = ['mse', 'rmse', 'mae'], True
        else:
            self.metric_list, self.spatial_norm = ['mse', 'mae'], False

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)

    def _init_distributed(self):
        """Initialize DDP training"""
        if self.args.fp16 and has_native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self.rank == 0:
               print('Using native PyTorch AMP. Training in mixed precision (fp16).')
        else:
            print('AMP not enabled. Training in float32.')
        self.model = NativeDDP(self.model, device_ids=[self.rank],
                               broadcast_buffers=self.args.broadcast_buffers,
                               find_unused_parameters=self.args.find_unused_parameters)

    def train_one_epoch(self, runner, train_loader, **kwargs): 
        """Train the model with train_loader.

        Args:
            runner: the trainer of methods.
            train_loader: dataloader of train.
        """
        raise NotImplementedError

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError

    def _dist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios in a distributed manner.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y,_ = self._predict(batch_x, batch_y)

            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_cat = np.concatenate([batch[k] for batch in results], axis=0)
            # gether tensors by GPU (it's no need to empty cache)
            results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
            results_all[k] = results_strip
        return results_all

    def _nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length
        eval_res = []

        for i, (batch_x, batch_y, batch_masks, batch_quantiles) in enumerate(data_loader):
            pred_y = torch.empty((batch_y.shape[0], len(data_loader.dataset.quantiles), batch_y.shape[1], batch_y.shape[2], batch_y.shape[3], batch_y.shape[4]))
            num_quantiles = len(data_loader.dataset.quantiles)
            with torch.no_grad():
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_quantiles = batch_quantiles.to(self.device)

                # Calculate intervals and assign predictions
                num_intervals = num_quantiles // 2  # Assuming symmetric intervals

                for j in range(num_intervals):
                    # Calculate interval from outer to inner quantiles
                    lower_idx = j
                    upper_idx = num_quantiles - 1 - j  # Convert to positive indexing
                    interval = (batch_quantiles[:, upper_idx:upper_idx+1] -
                            batch_quantiles[:, lower_idx:lower_idx+1])
                    # Get predictions for this interval
                    interval_pred, _ = self._predict([batch_x, interval], batch_y)

                    # Assign predictions directly
                    pred_y[:, j:j+1, ...] = interval_pred[:, 0:1, ...]  # Lower bound
                    pred_y[:, upper_idx:upper_idx+1, ...] = interval_pred[:, 2:3, ...]
                pred_y[:, num_quantiles//2:(num_quantiles//2)+1, ...] = interval_pred[:, 1:2, ...]
            # # combine the 3 predictions at a new dimension at axis 1

                #pred_y = torch.cat((pred_y_90, pred_y_60, pred_y_50), dim=1)




            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues', 'masks'],
                                        [batch_x[:,:,:,:,:].cpu().numpy(),
                                    pred_y[:,:,:,:,:,:].cpu().numpy()*batch_masks.unsqueeze(1).cpu().numpy(),
                                    batch_y[:,:,:,:,:].cpu().numpy(),
                                    batch_masks.cpu().numpy()])))
            else:  # return metrics
                eval_res = {}
                eval_res['train_loss'],eval_res['total_loss'],eval_res['mse'],eval_res['div'],eval_res['div_std'],eval_res['std'], eval_res['sum'] = self.criterion(pred_y, batch_y)
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].cpu().numpy().reshape(1)
                results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        # results_all['inputs'] are of shape (batch_size, 12, 1, 128, 128), set preds to be the average along the first dimension and duplicate it 12 times
        #preds = results_all['inputs'].mean(axis=1).repeat(12, 1)
        # set preds to zeros
        #preds = torch.zeros_like(torch.tensor(results_all['preds']))
        # make preds have an empty axis in 2nd dimension
        #preds = torch.tensor(np.expand_dims(preds, axis=2))


        #preds = torch.tensor(results_all['preds'])
        #results['trues'] = results['trues'][:,0:1,4:5,70,65]
        preds = torch.tensor(results_all['preds'])
        # clip preds to be between -255 and 255
        #preds = torch.clamp(preds, -255, 255)
        trues = torch.tensor(results_all['trues'])
        #losses_m = self.criterion_cpu(preds, trues)
        masks = torch.tensor(results_all['masks'])

        # create quantiles tensor of batch_sizex2 with static_ch.shape[0] as batch_size and 2 channels where the first is always 0.05 and the second is always 0.95
        quantiles = torch.tensor(data_loader.dataset.quantiles, dtype=torch.float).repeat(masks.shape[0], 1)
        
        pinball_loss, pinball_losses = self.val_criterion(preds[:,:,:], trues[:,:,:], masks[:,:,:], quantiles, train_run=False, loss_type='quantile')
        mis_loss, mis_losses = self.val_criterion(preds[:,:,:], trues[:,:,:], masks[:,:,:], quantiles, train_run=False, loss_type='mis')

        middle_index = quantiles.shape[1] // 2
        # compute the MSE of the total variation computed across the time dimension
        total_variation_preds = torch.abs(preds[:,middle_index,1:,:,:,:] - preds[:,middle_index,:-1,:,:,:])
        total_variation_trues = torch.abs(trues[:,1:,:,:,:] - trues[:,:-1,:,:,:])
        # sum the total variation over the time dimension
        total_variation_preds = total_variation_preds.sum(dim=1)
        total_variation_trues = total_variation_trues.sum(dim=1)
        mse_total_variation = MSE(total_variation_preds.cpu().numpy(), total_variation_trues.cpu().numpy())
        mae = MAE(preds[:,middle_index,:,:,:].squeeze().cpu().numpy(), trues.squeeze().cpu().numpy())
        mse = MSE(preds[:,middle_index,:,:,:].squeeze().cpu().numpy(), trues.squeeze().cpu().numpy())


        coverages = []
        mils = []

        num_quantiles = len(data_loader.dataset.quantiles)
        for i in range(num_quantiles // 2):
            lo_idx = i
            hi_idx = num_quantiles - 1 - i

            # TODO check trues.shape[1] is the appropriate time_step
            coverage, mil = eval_quantiles(preds[:, lo_idx], preds[:, hi_idx], trues, masks, time_step=trues.shape[1])
            coverages.append(coverage)
            mils.append(mil)

        results_all["pinball_loss"] = pinball_loss
        results_all["mis_loss"] = mis_loss
        results_all["pinball_losses"] = pinball_losses
        results_all["winkler_scores"] = mis_losses
        results_all["mae"] = mae
        results_all["mse"] = mse
        results_all["coverages"] = coverages
        results_all["mils"] = mils
        results_all["mse_total_variation"] = mse_total_variation

        return results_all

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            runner: the trainer of methods.
            val_loader: dataloader of validation.

        Returns:
            list(tensor, ...): The list of predictions and losses.
            eval_log(str): The string of metrics.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)
        else:
            results = self._nondist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=True)

        # eval_log = ""
        # for k, v in results.items():
        #     v = v.mean()
        #     if k != "loss":
        #         eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
        #         eval_log += eval_str

        return results

    def test_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(test_loader, gather_data=True)
        else:
            results = self._nondist_forward_collect(test_loader, gather_data=True)

        return results
    def grads_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(test_loader, gather_data=True)
        else:
            results = self._nondist_forward_grad(test_loader, gather_data=True)

        return results

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def clip_grads(self, params, norm_type: float = 2.0):
        """ Dispatch to gradient clipping method

        Args:
            parameters (Iterable): model parameters to clip
            value (float): clipping value/factor/norm, mode dependant
            mode (str): clipping mode, one of 'norm', 'value', 'agc'
            norm_type (float): p-norm, default 2.0
        """
        if self.clip_mode is None:
            return
        if self.clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
        elif self.clip_mode == 'value':
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        elif self.clip_mode == 'agc':
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({self.clip_mode})."
