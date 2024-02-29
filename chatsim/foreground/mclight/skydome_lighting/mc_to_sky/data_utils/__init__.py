from mc_to_sky.data_utils.hdr_sky_dataset import HDRSkyDataset
from mc_to_sky.data_utils.holicity_sdr_dataset import HoliCitySDRDataset
from mc_to_sky.data_utils.holicity_sdr_centered_dataset import HoliCitySDRCenteredDataset

def build_dataset(hypes, split):
    dataset_args = hypes['dataset']
    dataset_cls = eval(dataset_args['name'])
    return dataset_cls(dataset_args, split)
