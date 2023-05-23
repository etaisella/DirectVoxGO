_base_ = '../default.py'

expname = 'dvgo_lego2'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/lego2',
    dataset_type='vox-e',
    white_bkgd=True,
)

