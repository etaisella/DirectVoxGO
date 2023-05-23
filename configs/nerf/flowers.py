_base_ = '../default.py'

expname = 'dvgo_lego'
basedir = './logs/nerf_360'

data = dict(
    datadir='./data/nerf_360/flower',
    dataset_type='nerf_360',
    white_bkgd=True,
)

