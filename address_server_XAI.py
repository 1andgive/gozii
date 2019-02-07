# file address scripts
import platform
import os

addr_dataroot=os.path.join('codes', 'tools', 'data')

# sys.path.append('D:\\VQA\\BAN')
# BAN address
if(platform.system() == 'Linux'):
    addr_BAN = '../../VQA/BAN'
    addr_test_imgs='../../Data_Share/Datas/VQA_COCO/Images/test2015'
    addr_hdf5path='../../Data_Share/Datas/VQA_COCO/BottomUpPreTrain/hdf5'
    addr_coco_cap_train_path='../../Data_Share/Datas/VQA_COCO/annotations/captions_train2014.json'
    addr_coco_cap_val_path = '../../Data_Share/Datas/VQA_COCO/annotations/captions_val2014.json'
    
elif(platform.system() == 'Windows'):
    addr_BAN = 'D:\\VQA\\BAN'
    addr_test_imgs='D:\\Data_Share\\Datas\\VQA_COCO\\Images\\test2015'
    addr_hdf5path='../../Data_Share/Datas/VQA_COCO/BottomUpPreTrain/hdf5'
    addr_coco_cap_train_path='D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\captions_train2014.json'
    addr_coco_cap_val_path = 'D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\captions_val2014.json'
