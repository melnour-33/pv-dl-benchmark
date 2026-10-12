import os
import hashlib
import shelve
import dbm.dumb
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from models.arima_worker import _fit_forecast_one


class Model(nn.Module):

    def __init__(self, configs, order=None):
        """
        Per-window ARIMA baseline.

        Speed-ups relative to the serial version (forecast values unchanged):
          * windows of a batch are fitted in parallel (joblib, all cores);
          * every window is fitted ONCE with the longest horizon
            (arima_max_horizon, default 96) and the forecast is cached on disk.
            ARIMA multi-step forecasts are recursive, so the first H steps of a
            96-step forecast equal an H-step forecast. The runs for
            pred_len = 24/48/72 then read the cache instead of refitting.
          * the cache is written after every batch, so an interrupted run
            resumes where it stopped.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.features = getattr(configs, 'features', 'S')  # M, S, or MS
        self.order = order if order is not None else getattr(configs, 'arima_order', None)
        self.n_jobs = getattr(configs, 'arima_n_jobs', -1)
        self.max_h = max(int(getattr(configs, 'arima_max_horizon', 96)), self.pred_len)

        # disk cache: one file per dataset / feature mode / order / horizon
        cache_dir = getattr(configs, 'arima_cache_dir', './arima_cache')
        os.makedirs(cache_dir, exist_ok=True)
        order_tag = 'auto' if self.order is None else '_'.join(map(str, self.order))
        data_tag = getattr(configs, 'data', 'data')
        self.cache_path = os.path.join(
            cache_dir,
            f'{data_tag}_{self.features}_sl{self.seq_len}_H{self.max_h}_{order_tag}')

        # dummy parameter so the shared training/checkpoint code has a parameter
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    @staticmethod
    def _key(series, exog):
        h = hashlib.sha1(np.ascontiguousarray(series, dtype=np.float32).tobytes())
        if exog is not None:
            h.update(np.ascontiguousarray(exog, dtype=np.float32).tobytes())
        return h.hexdigest()

    def _run_jobs(self, jobs):
        """jobs: list of (series, exog). Returns list of forecasts (len max_h)."""
        keys = [self._key(s, e) for s, e in jobs]
        out = [None] * len(jobs)
        # dbm.dumb: pure-Python, no file locking -> works on Lustre/GPFS and Windows
        with shelve.Shelf(dbm.dumb.open(self.cache_path, 'c')) as db:
            todo = []
            for i, k in enumerate(keys):
                if k in db:
                    out[i] = db[k]
                else:
                    todo.append(i)
            if todo:
                res = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(_fit_forecast_one)(jobs[i][0], jobs[i][1], self.order, self.max_h)
                    for i in todo)
                for i, f in zip(todo, res):
                    out[i] = f
                    db[keys[i]] = f
        return out

    def forecast(self, x_enc):
        batch_size, seq_len, num_channels = x_enc.shape
        device = x_enc.device
        x_np = x_enc.detach().cpu().numpy()
        forecasts = np.zeros((batch_size, self.pred_len, num_channels))

        if self.features == 'MS':
            # target = last channel; the other channels are exogenous regressors
            tc = num_channels - 1
            jobs = [(x_np[b, :, tc], x_np[b, :, :tc] if tc > 0 else None)
                    for b in range(batch_size)]
            res = self._run_jobs(jobs)
            for b in range(batch_size):
                forecasts[b, :, tc] = res[b][:self.pred_len]
        else:
            # M or S: independent univariate ARIMA per channel per window
            jobs = [(x_np[b, :, c], None)
                    for b in range(batch_size) for c in range(num_channels)]
            res = self._run_jobs(jobs)
            j = 0
            for b in range(batch_size):
                for c in range(num_channels):
                    forecasts[b, :, c] = res[j][:self.pred_len]
                    j += 1

        forecasts_tensor = torch.tensor(forecasts, dtype=torch.float32, device=device)
        # zero-scaled dummy keeps the output in the autograd graph (values unchanged)
        return forecasts_tensor + self.dummy_param.expand_as(forecasts_tensor) * 0.0

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc)
        raise ValueError(f'ARIMA model only supports forecast tasks, got {self.task_name}')
