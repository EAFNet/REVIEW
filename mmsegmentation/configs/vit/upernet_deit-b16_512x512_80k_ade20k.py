_base_ = './upernet_vit-b16_mln_512x512_80k_ade20k.py'

model = dict(
    pretrained='/home/sdxx/桌面/mmsegmentation-master/upernet_deit-b16_512x512_80k_ade20k_20210624_130529-1e090789.pth',
    backbone=dict(drop_path_rate=0.1),
    neck=None)
