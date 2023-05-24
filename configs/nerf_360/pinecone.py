_base_ = '../default.py'

expname = 'dvgo_pinecone'
basedir = './logs/nerf_360/pinecone'

data = dict(
    datadir='./data/nerf_360/pinecone',
    dataset_type='vox-e',
    white_bkgd=True,
)

