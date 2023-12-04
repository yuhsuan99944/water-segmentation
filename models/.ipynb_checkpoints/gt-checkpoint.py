from ml4floods.models.config_setup import get_default_config
from ml4floods.models.dataset_setup import get_dataset
import pkg_resources

# Set filepath to configuration files
# config_fp = 'path/to/worldfloods_template.json'
config_fp = pkg_resources.resource_filename("ml4floods","models/configurations/worldfloods_template.json")
# 模型名稱
config = get_default_config(config_fp)
config.data_params.target_folder = 'gt'
dataset = get_dataset(config.data_params)

def gt_batch(dataset):
    train_dl = dataset.train_dataloader()
    train_dl_iter = iter(train_dl)
    batch_train = next(train_dl_iter)

    val_dl = dataset.val_dataloader()
    val_dl_iter = iter(val_dl)
    batch_val = next(val_dl_iter)

    test_dl = dataset.test_dataloader()
    test_dl_iter = iter(test_dl)
    batch_test = next(test_dl_iter)

    # gt = {"train": batch_train['mask'], "val":batch_val['mask'], "test": batch_test['mask']}
    gt = {"train": batch_train['mask'], "val":batch_val['mask']}
    return gt