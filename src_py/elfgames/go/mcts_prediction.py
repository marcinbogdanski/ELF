# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.autograd import Variable

import elf.logging as logging
from elf.options import auto_import_options, PyOptionSpec
from rlpytorch.trainer.timer import RLTimer


class MCTSPrediction(object):
    @classmethod
    def get_option_spec(cls):
        spec = PyOptionSpec()
        spec.addBoolOption(
            'backprop',
            'Whether to backprop the total loss',
            True)
        return spec

    @auto_import_options
    def __init__(self, option_map):
        self.policy_loss = nn.KLDivLoss().cuda()
        self.value_loss = nn.MSELoss().cuda()
        self.logger = logging.getIndexedLogger(
            'elfgames.go.MCTSPrediction-', '')
        self.timer = RLTimer()

    def update(self, mi, batch, stats, use_cooldown=False, cooldown_count=0):   # mi=rlpytorch.model_interface.ModelInterface batch=elf.utils_elf.Batch stats={} use_cooldown=False, cooldown_count=0
        ''' Update given batch '''                                              # batch.batch['s'] tensor [2048, 18, 9, 9], float32, cuda, min=0, max=1,  mean= 0.3277, std= 0.4694
        self.timer.restart()                                                    # batch.batch['offline_a'], t, [2048, 1],   int64,   cuda, min=0, max=81, mean=40.2583, std=24.5416
        if use_cooldown:                                                        # batch.batch['winner']     t, [2048],      float32, cuda, min=-1, max=1, mean= 0.0859, std= 0.9965
            if cooldown_count == 0:                                             # batch.batch['mcts_scores'], t[2048, 83],  float32, cuda, min=0, max=1,  mean= 0.0122, std= 0.0378     # rows add to 1
                mi['model'].prepare_cooldown()                                  # batch.batch['move_idx'], t,  [2048],      int32,   cuda, min=0, max=160,mean=77.9287, std=45.9748
                self.timer.record('prepare_cooldown')                           # batch.batch['selfplay_ver']t [2048],      int64,   cuda, min=0, max=0,  mean=0        std=0

        # Current timestep.
        state_curr = mi['model'](batch)                                         # state_curr['logpi'] t        [2048, 82],  float32, cuda, min=-5.4212, max=-3.5183, mean=-4.4308, std=0.2204
        self.timer.record('forward')                                            # state_curr['pi'] t           [2048, 82],  float32, cuda, min= 0.0044, max= 0.0296, mean= 0.0122, std=0.0027  # rows sum to 1
                                                                                # state_curr['V'] t            [2048, 1],   float32, cuda, min=-0.1441, max= 0.2090, mean=-0.0059, std=0.0426
        if use_cooldown:
            self.logger.debug(self.timer.print(1))
            return dict(backprop=False)

        targets = batch["mcts_scores"]                                          # ^ t   [2048, 83],  float32, cuda, min=0, max=1,  mean= 0.0122, std= 0.0378     # rows add to 1
        logpi = state_curr["logpi"]                                             # ^ t   [2048, 82],  float32, cuda, min=-5.4212, max=-3.5183, mean=-4.4308, std=0.2204
        pi = state_curr["pi"]                                                   # ^ t   [2048, 82],  float32, cuda, min= 0.0044, max= 0.0296, mean= 0.0122, std=0.0027  # rows sum to 1
        # backward.
        # loss = self.policy_loss(logpi, Variable(targets)) * logpi.size(1)
        loss = - (logpi * Variable(targets)                                     # generic formulation of CE = -sum(y * log(y_hat)), y is target vector of probabilities, y_hat is predicted
                  ).sum(dim=1).mean()  # * logpi.size(1)                        # loss=4.4350
        stats["loss"].feed(float(loss))
        total_policy_loss = loss

        entropy = (logpi * pi).sum() * -1 / logpi.size(0)                       # entropy=4.3828
        stats["entropy"].feed(float(entropy))

        stats["blackwin"].feed(                                                 # 0.5429
            float((batch["winner"] > 0.0).float().sum()) /
            batch["winner"].size(0))

        total_value_loss = None                                                 # 1.0031
        if "V" in state_curr and "winner" in batch:
            total_value_loss = self.value_loss(
                state_curr["V"].squeeze(), Variable(batch["winner"]))

        stats["total_policy_loss"].feed(float(total_policy_loss))
        if total_value_loss is not None:
            stats["total_value_loss"].feed(float(total_value_loss))
            total_loss = total_policy_loss + total_value_loss                   # 5.4381
        else:
            total_loss = total_policy_loss

        stats["total_loss"].feed(float(total_loss))
        self.timer.record('feed_stats')

        if self.options.backprop:
            total_loss.backward()
            self.timer.record('backward')
            self.logger.debug(self.timer.print(1))
            return dict(backprop=True)
        else:
            self.logger.debug(self.timer.print(1))
            return dict(backprop=False)
