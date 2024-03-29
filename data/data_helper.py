from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import build_transform
from data.JigsawLoader import JigsawDatasetRandAug, JigsawTestDatasetRandAug, get_split_domain_info_from_dir, \
    get_split_dataset_info_from_txt, _dataset_info
from data.concat_dataset import ConcatDataset
from data.JigsawLoader import JigsawNewDataset, JigsawTestNewDataset

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
officehome_datasets = ['Art', 'Clipart', 'Product', 'RealWorld']
digits_datasets = ["mnist", "mnist_m", "svhn", "syn"]
available_datasets = officehome_datasets + pacs_datasets + vlcs_datasets + digits_datasets


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args, patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []

    # infer_no_resize = False
    # img_transformer = build_transform(True, args, infer_no_resize)
    # img_transformer_val = build_transform(False, args, infer_no_resize)
    # _, tile_transformer = get_train_transformers(args)

    img_transformer, tile_transformer = get_train_transformers(args)
    img_transformer_val = get_val_transformer(args)

    limit = args.limit_source
    if "PACS" in args.data_path:
        dataset_path = join(args.data_path, "kfold")
    elif args.data == "miniDomainNet":
        dataset_path = "/data/DataSets/" + "DomainNet"
    else:
        dataset_path = args.data_path

    for i, dname in enumerate(dataset_list):
        if args.data == "PACS":
            name_train, name_val, labels_train, labels_val, domain_labels_train, domain_labels_val = \
                get_split_dataset_info_from_txt(txt_path=join(args.data_path, "pacs_label"), domain=dname,
                                                domain_label=i+1)
        elif args.data == "miniDomainNet":
            name_train, name_val, labels_train, labels_val, domain_labels_train, domain_labels_val = \
                get_split_dataset_info_from_txt(txt_path=args.data_root, domain=dname, domain_label=i+1,
                                                val_percentage=args.val_size)
        else:
            name_train, name_val, labels_train, labels_val, domain_labels_train, domain_labels_val = \
                get_split_domain_info_from_dir(join(dataset_path, dname), dataset_name=args.data,
                                               val_percentage=args.val_size, domain_label=i+1)

        if args.RandAug_flag == 1:
            train_dataset = JigsawDatasetRandAug(name_train, labels_train, domain_labels_train,
                                                 dataset_path=dataset_path, patches=patches,
                                                 img_transformer=img_transformer,
                                                 bias_whole_image=args.bias_whole_image, args=args)
        else:
            train_dataset = JigsawNewDataset(name_train, labels_train, domain_labels_train,
                                             dataset_path=dataset_path, patches=patches,
                                             img_transformer=img_transformer, tile_transformer=tile_transformer,
                                             jig_classes=30, bias_whole_image=args.bias_whole_image)
        if limit:
            train_dataset = Subset(train_dataset, limit)
        datasets.append(train_dataset)
        val_datasets.append(
            JigsawTestNewDataset(name_val, labels_val, domain_labels_val, dataset_path=dataset_path,
                                 img_transformer=img_transformer_val, patches=patches, jig_classes=30))
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    #                                      pin_memory=True, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
    #                                          pin_memory=True, drop_last=False)
    # return loader, val_loader
    return dataset, val_dataset


def get_val_dataloader(args, patches=False, tSNE_flag=0):
    if "PACS" in args.data_path:
        dataset_path = join(args.data_path, "kfold")
    elif args.data == "miniDomainNet":
        dataset_path = "/data/DataSets/" + "DomainNet"
    else:
        dataset_path = args.data_path

    if args.data == "miniDomainNet":
        name_train, name_val, labels_train, labels_val, domain_label_train, domain_label_val = \
            get_split_dataset_info_from_txt(txt_path=args.data_root, domain=args.target, domain_label=0,
                                            val_percentage=args.val_size)
    else:
        name_train, name_val, labels_train, labels_val, domain_label_train, domain_label_val =\
            get_split_domain_info_from_dir(
            join(dataset_path, args.target), dataset_name=args.data, val_percentage=args.val_size, domain_label=0)

    if tSNE_flag == 0:
        names = name_train + name_val
        labels = labels_train + labels_val
        domain_label = domain_label_train + domain_label_val
    else:
        names = name_val
        labels = labels_val
        domain_label = domain_label_val

   # img_tr =  build_transform(False, args, False)  #我修改的，为了使用GFNet的原始版本数据处理
    img_tr = get_val_transformer(args)

    val_dataset = JigsawTestNewDataset(names, labels, domain_label, dataset_path=dataset_path, patches=patches,
                                       img_transformer=img_tr, jig_classes=30)

    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)

    dataset = ConcatDataset([val_dataset])
    # loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
    #                                      pin_memory=True, drop_last=False)
    # return loader
    return dataset


def get_train_transformers(args):
    if args.data == 'OfficeHome':
        size = int((256 / 224) * args.input_size)
        img_tr = [transforms.Resize(size, interpolation=3)]
        img_tr.append(transforms.CenterCrop(args.input_size))   #这三行OH用，因为OH的图片大小不一致
  #始终不用这行  img_tr = [transforms.RandomResizedCrop((int(args.input_size), int(args.input_size)), (args.min_scale, args.max_scale))]
    else:
        img_tr = [transforms.Resize(args.input_size, interpolation=3)]  # PACS和VLCS和DIGITS用

    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    # this is special operation for JigenDG
    if args.gray_flag:
        img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))

    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    tile_tr = []
    if args.gray_flag:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(img_tr)
