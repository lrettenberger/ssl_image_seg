from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from DLIP.utils.callbacks.epoch_duration_log import EpochDurationLogCallback
from DLIP.utils.callbacks.log_best_metric import LogBestMetricsCallback

from DLIP.utils.callbacks.image_seg_log import ImageLogSegCallback
from DLIP.utils.callbacks.image_inst_seg_log import ImageLogInstSegCallback
from DLIP.utils.callbacks.log_instance_seg_metrics import LogInstSegMetricsCallback
from DLIP.utils.callbacks.increase_ssl_img_size import IncreaseSSLImageSizeCallback

from DLIP.utils.callbacks.unet_log_instance_seg_img import UNetLogInstSegImgCallback

from DLIP.utils.callbacks.detectron_log_instance_seg_metrics import DetectronLogInstSegMetricsCallback
from DLIP.utils.callbacks.detectron_log_instance_seg_img import DetectronLogInstSegImgCallback
from DLIP.utils.callbacks.detectron_log_sem_seg_img import DetectronLogSemSegImgCallback

from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.loading.dict_to_config import dict_to_config

class CallbackCompose:
    def __init__(
        self,
        params,
        data
    ):
        self.params = params
        self.callback_lst = None
        self.data = data
        self.make_composition()

    def make_composition(self):
        self.callback_lst = []

        if hasattr(self.params, 'save_k_top_models'):
            self.weights_dir = self.params.experiment_dir
            self.weights_name = 'dnn_weights'
            self.callback_lst.append(
                ModelCheckpoint(
                    filename = self.weights_name,
                    dirpath = self.weights_dir,
                    save_top_k=self.params.save_k_top_models,
                    monitor='val/loss'
                )
            )

        if hasattr(self.params, 'early_stopping_enabled') and self.params.early_stopping_enabled:
            self.callback_lst.append(
                EarlyStopping(
                monitor='val/loss',
                patience=self.params.early_stopping_patience,
                verbose=True,
                mode='min'
                )
            )

        if hasattr(self.params, 'epoch_duration_enabled') and self.params.epoch_duration_enabled:
            self.callback_lst.append(
                EpochDurationLogCallback()
            )

        if hasattr(self.params, 'best_metrics_log_enabled') and self.params.best_metrics_log_enabled:
            self.callback_lst.append(
                LogBestMetricsCallback(
                    self.params.log_best_metric_dict
                )
            )

        # UNet Callbacks
        if hasattr(self.params, 'img_seg_log_enabled')  and self.params.img_log_enabled:
            self.callback_lst.append(
                ImageLogSegCallback()
            )

        if hasattr(self.params, 'img_log_inst_seg_enabled')  and self.params.img_log_inst_seg_enabled:
            inst_seg_pp_params = split_parameters(dict_to_config(vars(self.params)), ["inst_seg_pp"])["inst_seg_pp"]
            self.callback_lst.append(
                ImageLogInstSegCallback(inst_seg_pp_params)
            )

        if hasattr(self.params, 'inst_seg_metrics_log_enabled') and self.params.inst_seg_metrics_log_enabled:
            inst_seg_pp_params = split_parameters(dict_to_config(vars(self.params)), ["inst_seg_pp"])["inst_seg_pp"]
            self.callback_lst.append(
                LogInstSegMetricsCallback(inst_seg_pp_params)
            )

        if hasattr(self.params, 'increase_ssl_image_size_enabled') and self.params.increase_ssl_image_size_enabled:
            factor = 2
            if hasattr(self.params, 'increase_ssl_image_size_factor'):
                factor = self.params.increase_ssl_image_size_factor
            self.callback_lst.append(
                IncreaseSSLImageSizeCallback(increase_factor=factor)
            )

        # UNet Callbacks
        if hasattr(self.params, 'unet_inst_seg_img_log_enabled') and self.params.unet_inst_seg_img_log_enabled:
            self.callback_lst.append(
                UNetLogInstSegImgCallback()
            )    
        

        # Detectron Callbacks
        if hasattr(self.params, 'detectron_inst_seg_metrics_log_enabled') and self.params.detectron_inst_seg_metrics_log_enabled:
            self.callback_lst.append(
                DetectronLogInstSegMetricsCallback()
            )    

        if hasattr(self.params, 'detectron_inst_seg_img_log_enabled') and self.params.detectron_inst_seg_img_log_enabled:
            self.callback_lst.append(
                DetectronLogInstSegImgCallback()
            )    

        if hasattr(self.params, 'detectron_sem_seg_img_log_enabled') and self.params.detectron_sem_seg_img_log_enabled:
            self.callback_lst.append(
                DetectronLogSemSegImgCallback()
            )    

    def get_composition(self):
        return self.callback_lst