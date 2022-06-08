import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from patent_phrase_similarity.models.deberta_pytorch import CFG
from utils.functions import KaggleFunctions as KF


class PPPMModel(nn.Module):
    def __init__(self, model_path="microsoft/mdeberta-v3-base", config_path=None, config_updates=None, reinitialization_layers=0, mixout=0.0):
        super(PPPMModel, self).__init__()
        self.config = CFG()
        self.model_config, self.model = self._prepare_model(model_path, config_path, config_updates=config_updates)
        self._reinitialize_layers(
            self.model.encoder.layer,
            n_layers=reinitialization_layers,
            std=self.model_config.initializer_range
        )

        # Top Layers
        # self.bert = AutoModel.from_pretrained(model_name)
        # self.head = nn.Linear(1024, 1, bias=True)
        # self.dropout = nn.Dropout(0.5)

        # self.head = nn.Linear(in_features=self.config.hidden_size, out_features=1)
        # self._init_weights(self.head, std=self.config.initializer_range)

        self.fc_dropout = nn.Dropout(self.config.model.fc_dropout)
        self.fc = nn.Linear(self.model_config.hidden_size, self.config.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention, std=self.model_config.initializer_range)

    @staticmethod
    def _prepare_model(model_path, config_path, config_updates=None):
        if config_path is None:
            _config = AutoConfig.from_pretrained(model_path)
        else:
            _config = AutoConfig.from_pretrained(config_path)

        if config_updates is None:
            config_updates = {}

        # Show hidden states
        _config.output_hidden_states = True
        _config.update(config_updates)

        if config_path is None:
            _model = AutoModel.from_pretrained(model_path, config=_config)
        else:
            _model = AutoModel.from_config(_config)

        return _config, _model

    @staticmethod
    def _init_weights(module, std=0.02):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _reinitialize_layers(self, layers, n_layers=0, std=0.02):
        if n_layers > 0:
            for layer in layers[-n_layers:]:
                for name, module in layer.named_modules():
                    self._init_weights(module, std=std)

            print(f"Reinitializated last {n_layers} layers.")

    #     def forward(self, inputs):
    #         outputs = self.model(**inputs)
    #         last_hidden_states = outputs[0]
    #         # feature = torch.mean(last_hidden_states, 1)
    #         weights = self.attention(last_hidden_states)
    #         feature = torch.sum(weights * last_hidden_states, dim=1)

    #         output = self.fc(self.fc_dropout(feature))
    #         return output
    #         transformer_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #         features = transformer_outputs.hidden_states[-1]
    #         features = features[:, 0, :]
    #         outputs = self.head(features)

    def forward(self, token_type_ids, attention_mask):
        outputs = self.model(token_type_ids, attention_mask)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        output = self.fc(self.fc_dropout(feature))
        # feature = self.model(token_type_ids, attention_mask)
        # feature = torch.sum(feature[0], 1)/feature[0].shape[1]
        # feature = self.fc_dropout(feature)
        # output = self.fc(feature)
        return output


class PPPMTrainerDefinition(pl.LightningModule):
    def __init__(self, model, criterion, metric, train_ds_size):
        super(PPPMTrainerDefinition, self).__init__()
        self.config = CFG()

        self.criterion = criterion
        self.metric = metric
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.apex)
        self.scaler = None
        self.train_ds_size = train_ds_size

        # Model Load
        self.model = model
        torch.save(self.model.config, 'model_config.pth')
        self.model.to(self.config.torch_device)

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # Total steps in this pipeline
        self.total_steps = None

    def setup(self, stage=None):
        if stage == 'fit':
            # Calculate total steps
            self.total_steps = (
                (self.train_ds_size // (self.config.batch_size * max(1, 1 if self.config.device == 'cuda' else 0))) //
                self.config.gradient_accumulation_batches *
                self.config.epochs
            )

    def configure_callbacks(self, checkpoint_callback=None, early_stop_callback=None):
        # checkpoint_callback = ModelCheckpoint(
        #     monitor=self.config.monitor,
        #     save_top_k=1,
        #     save_last=True,
        #     save_weights_only=True,
        #     filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
        #     verbose=False,
        #     mode='min'
        # )
        # early_stop_callback = EarlyStopping(
        #     monitor=self.config.monitor,
        #     patience=self.config.patience,
        #     verbose=False,
        #     mode="min"
        # )
        return []

    def configure_optimizers(self):
        def _get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
            # param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            _optimizer_parameters = [
                {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if "model" not in n],
                 'lr': decoder_lr, 'weight_decay': 0.0}
            ]
            return _optimizer_parameters

        def _get_scheduler(optimizer, num_train_steps):
            _scheduler = None
            if self.config.scheduler == 'linear':
                _scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=self.config.num_warmup_steps, num_training_steps=num_train_steps
                )
            elif self.config.scheduler == 'cosine':
                _scheduler = get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps=self.config.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=self.config.num_cycles
                )
            return _scheduler

        # Optimzers
        optimizer_model_parameters = _get_optimizer_params(
            self.model,
            encoder_lr=self.config.encoder_lr,
            decoder_lr=self.config.decoder_lr,
            weight_decay=self.config.optimizer.parameters.weight_decay
        )

        optimizer = KF.get_optimizer(
            self.config.optimizer.name,
            model_parameters=optimizer_model_parameters,
            parameters=dict(lr=self.config.encoder_lr, eps=self.config.eps, betas=self.config.betas)
        )
        # Schedulers
        scheduler = _get_scheduler(optimizer, self.total_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def forward(self, token_type_ids, attention_mask):
        logits = self.model(token_type_ids, attention_mask)
        return logits

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        token_type_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        token_type_ids.to(self.config.torch_device)
        attention_mask.to(self.config.torch_device)
        labels.to(self.config.torch_device)

        with torch.cuda.amp.autocast(enabled=self.config.apex):
            y_preds = self.model(token_type_ids, attention_mask)
        loss = self.criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        rmse = self.metric(y_preds.squeeze(1), labels)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        if not self.automatic_optimization:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                self.manual_backward(loss)

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                if self.config.batch_scheduler:
                    scheduler.step()

        logs = {
            'train_loss': loss,
            'train_error': rmse,
            # 'lr': self.optimizer.param_groups[0]['lr'],
            'lr': scheduler.get_lr()[0],
            'grad_norm': grad_norm
        }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return logs

    # def training_step(self, batch, batch_idx):
    #     token_type_ids, attention_mask = batch[0], batch[1]
    #     preds = self.model(token_type_ids, attention_mask)
    #     loss = self.criterion(preds.squeeze(1), batch[2])
    #     rmse = self.metric(preds.squeeze(1), batch[2])
    #     logs = {'train_loss': loss, 'train_error': rmse, 'lr': self.optimizer.param_groups[0]['lr']}
    #     self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    def training_epoch_end(self, outputs):
        avg_loss = np.mean([x['train_loss'] for x in outputs])
        avg_error = np.mean([x['train_error'] for x in outputs])
        logs = {'avg_loss': avg_loss, 'avg_error': avg_error}
        self.log_dict(logs, on_epoch=True, prog_bar=True, logger=True)
        return {**logs, 'log': logs}

    def validation_step(self, batch, batch_idx):
        token_type_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        with torch.no_grad():
            y_preds = self.model(token_type_ids, attention_mask)

        loss = self.criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        rmse = self.metric(y_preds.squeeze(1), labels)
        score = KF.get_pearsonr_score(labels, y_preds.squeeze(1))

        logs = {'val_loss': loss, 'val_error': rmse, 'val_score': score}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return logs

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([x['val_loss'] for x in outputs])
        avg_error = np.mean([x['val_error'] for x in outputs])
        avg_score = np.mean([x['val_score'] for x in outputs])
        logs = {'val_avg_loss': avg_loss, 'val_avg_error': avg_error, 'val_avg_score': avg_score}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {**logs, 'log': logs}

    def test_step(self, batch, batch_idx):
        token_type_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        with torch.no_grad():
            y_preds = self.model(token_type_ids, attention_mask)

        loss = self.criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        rmse = self.metric(y_preds.squeeze(1), batch[2])
        score = KF.get_pearsonr_score(labels, y_preds.squeeze(1))

        logs = {'test_loss': loss, 'test_error': rmse, 'test_score': score}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return logs

    def test_epoch_end(self, outputs):
        avg_loss = np.mean([x['val_loss'] for x in outputs])
        avg_error = np.mean([x['val_error'] for x in outputs])
        avg_score = np.mean([x['val_score'] for x in outputs])
        logs = {'val_avg_loss': avg_loss, 'val_avg_error': avg_error, 'val_avg_score': avg_score}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {**logs, 'log': logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        token_type_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        # self.model.eval()
        self.model.to(self.config.torch_device)
        with torch.no_grad():
            y_preds = self.model(token_type_ids, attention_mask)
            y_preds.sigmoid().to(self.config.torch_device).numpy()

        return y_preds