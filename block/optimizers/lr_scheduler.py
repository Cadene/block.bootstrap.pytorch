import torch
import torch.nn as nn
from bootstrap.lib.logger import Logger

class ReduceLROnPlateau():

    def __init__(self,
            optimizer,
            engine=None,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=False,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08):
        self.optimizer = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            verbose=verbose,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps)
        if engine is not None:
            engine.register_hook('train_on_begin_epoch', self.step_lr_scheduler)
            #engine.register_hook('eval_on_end_epoch', self.step_lr_scheduler)

    def step_lr_scheduler(self):
        prev_lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
        Logger().log_value('train_epoch.lr', prev_lr)

        # first epoch
        if len(Logger().values['eval_epoch.loss']) == 0:
            return

        val_loss = Logger().values['eval_epoch.loss'][-1]
        self.lr_scheduler.step(val_loss)
        new_lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            Logger()('ReduceLROnPlateau: lr has changed from {} to {}'.format(prev_lr, new_lr))

    def __getattr__(self, key):
        try:
            return super(ReduceLROnPlateau, self).__getattr__(key)
        except AttributeError:
            return self.optimizer.__getattribute__(key)


# Inspired from https://github.com/jnhwkim/ban-vqa/blob/master/train.py
class BanOptimizer():

    def __init__(self, engine,
            name='Adamax',
            lr=0.0007,
            gradual_warmup_steps=[0.5, 2.0, 4],
            lr_decay_epochs=[10, 20, 2],
            lr_decay_rate=.25):
        self.engine = engine
        self.optimizer = torch.optim.__dict__[name](
            filter(lambda p: p.requires_grad, engine.model.network.parameters()),
            lr=lr
        )
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = eval("range({},{},{})".format(*lr_decay_epochs))

        self.gradual_warmup_steps = [
            weight * lr for weight in eval("torch.linspace({},{},{})".format(
                gradual_warmup_steps[0],
                gradual_warmup_steps[1],
                int(gradual_warmup_steps[2])
            ))
        ]
        self.grad_clip = .25
        self.total_norm = 0
        self.count_norm = 0
        if engine:
            engine.register_hook('train_on_start_epoch', self.set_lr)
            engine.register_hook('train_on_print', self.display_norm)

    def set_lr(self):
        epoch_id = self.engine.epoch
        optim = self.optimizer
        old_lr = optim.param_groups[0]['lr']
        if epoch_id < len(self.gradual_warmup_steps):
            new_lr = self.gradual_warmup_steps[epoch_id]
            optim.param_groups[0]['lr'] = new_lr
            Logger()('Gradual Warmup lr: {:.8f} -> {:.8f}'.format(old_lr, new_lr))
        elif epoch_id in self.lr_decay_epochs:
            new_lr = optim.param_groups[0]['lr'] * self.lr_decay_rate
            optim.param_groups[0]['lr'] = new_lr
            Logger()('Decrease lr: {:.8f} -> {:.8f}'.format(old_lr, new_lr))
        else:
            Logger()('No change to lr: {:.8f}'.format(old_lr))
        Logger().log_value('train_epoch.lr', optim.param_groups[0]['lr'].item())

    def display_norm(self):
        Logger()('      norm: {:.5f}'.format(self.total_norm / self.count_norm))

    def step(self):
        self.total_norm += nn.utils.clip_grad_norm_(
            self.engine.model.network.parameters(),
            self.grad_clip
        )
        self.count_norm += 1
        self.optimizer.step()
        Logger().log_value('train_batch.norm', self.total_norm / self.count_norm)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        state = {}
        state['optimizer'] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])

    def __getattr__(self, key):
        try:
            return super(ReduceLROnPlateau, self).__getattr__(key)
        except AttributeError:
            return self.optimizer.__getattribute__(key)
