

def get_dataset_info(dataset_type):
    """define dataset_name and its dataroot"""
    root = ''
    if dataset_type == 'LEVIR_CD':
        root = 'path-to-LEVIR_CD-dataroot'
    # add more dataset ...
    else:
        raise TypeError("not define the %s" % dataset_type)

    return root
