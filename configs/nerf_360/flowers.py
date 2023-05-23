_base_ = '../default.py'

expname = 'dvgo_flowers'
basedir = './logs/nerf_360/flowers'

data = dict(
    datadir='./data/nerf_360/flowers',
    dataset_type='vox-e',
    white_bkgd=True,
)

