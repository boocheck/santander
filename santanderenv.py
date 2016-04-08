import platform

__config = [
    {"name": "work_quest", "node": "boochecBox", "type": "g", "data": "/home/boocheck/datasets", "train": "/home/boocheck/train"},
    {"name": "work_host", "node": "LIS-BUCZKOWSKI", "type": "h", "data-train": "D:\\datasets\\santander\\train.csv", "data-test": "D:\\datasets\\santander\\test.csv", "results": "D:\\results\\santander"},
    {"name": "spark", "node": "spark2.opi.org.pl", "type": "s", "data": "/home/boocheck/datasets", "train": "/home/boocheck/train"},
    {"name": "home_host", "node": "PrzemoLap", "type": "h", "data-train": "D:\\Datasets\\santander\\train.csv", "data-test": "D:\\Datasets\\santander\\test.csv", "results": "D:\\results\\santander"},
    {"name": "home_guest", "node": "devbuntu", "type": "g", "data": "/media/sf_Datasets/steam", "train": "/media/sf_Coding/python/steamflow/train"}
]

def get_config(key=None):
    res_config = filter(lambda x: x["node"] == platform.node(), __config)[0]
    if(key != None):
        return res_config[key]
    else:
        return res_config