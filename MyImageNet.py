import warnings
from contextlib import contextmanager
import os
import shutil
import tempfile
from typing import Any, Dict, List, Iterator, Optional, Tuple
import torch
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, extract_archive, verify_str_arg

"""
This module is to extract ImageNet dataset. 
Two folders:
  -- <folderPath>/train
  -- <folderPath>/val
Three files:
  -- <filePath>/ILSVRC2012_validation_ground_truth.txt
  -- <filePath>/LOC_synset_mapping.txt
  -- <filePath>/meta.bin
"""

ARCHIVE_META = {
    'train': 'train',
    'val': 'val',
    'devkit': './data/ImageNet/ILSVRC/LOC_synset_mapping.txt'
}


META_FILE = "./data/ImageNet/ILSVRC/meta.bin"


class MyImageNet(ImageFolder):
    """`ImageNet'_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the two folders of imageSet.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: str = 'train', download: Optional[str] = None, **kwargs: Any) -> None:

        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super(MyImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnid_to_classes.keys())}
        # idx_to_targets = {idx: wnid_to_idx[wnid] for idx, wnid in enumerate(self.wnids)}
        self.wnid_to_idx = self.class_to_idx
        # self.wnid_to_idx = {wnid: wnid_to_idx[wnid] for wnid in self.wnids}
        # self.target_transform = lambda idx: idx_to_targets[idx]
        # self.samples = self.samples
        # self.targets = [idx_to_targets[idx] for idx in self.targets]
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: self.wnid_to_idx[wnid]
                             for wnid in self.wnids
                             for cls in wnid_to_classes[wnid]}

    def parse_archives(self) -> None:
        if not check_integrity(os.path.expanduser(META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)
      
    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.expanduser(file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is corrupted. "
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))


def _verify_archive(root: str, file: str) -> None:
    if not check_integrity(os.path.join(root, file)):
        msg = ("The archive {} is not present in the root directory or is corrupted. "
               "You need to download it externally and place it in {}.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root: str, file: Optional[str] = None) -> None:
    """Parse the loc synset mapping file of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """

    def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, str]]:

        if not os.path.isfile(devkit_root):
            print('no loc synset mapping file found')
        else:
            wnids = []
            classes = []
            with open(devkit_root, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    x1, x2 = line.rstrip().split(' ', 1)
                    wnids.append(x1)
                    classes.append(x2)

        idcs = range(len(wnids))
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root,
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir() -> Iterator[str]:
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ARCHIVE_META["devkit"]
    if file is None:
        file = archive_meta

    if os.path.exists(os.path.join(root, file)):
        # extract_archive(os.path.join(root, file), tmp_dir)
        devkit_root = os.path.join(root, file)
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))

    else:
        print('Annotation or label data not found')




def parse_train_archive(root: str, file: Optional[str] = None, folder: str = "train") -> None:
    """Parse the train images archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the train images archive
        file (str, optional): Name of train images archive. Defaults to
            'ILSVRC2012_img_train.tar'
        folder (str, optional): Optional name for train images folder. Defaults to
            'train'
    """
    archive_meta = ARCHIVE_META["train"]
    if file is None:
        file = archive_meta

    train_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), train_root)

    archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
    for archive in archives:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def parse_val_archive(
    root: str, file: Optional[str] = None, wnids: Optional[List[str]] = None, folder: str = "val"
) -> None:
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the validation images archive
        file (str, optional): Name of validation images archive. Defaults to
            'ILSVRC2012_img_val.tar'
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are loaded from the meta file in the root directory
        folder (str, optional): Optional name for validation images folder. Defaults to
            'val'
    """
    archive_meta = ARCHIVE_META["val"]
    if file is None:
        file = archive_meta
    # md5 = archive_meta[1]
    if wnids is None:
        wnids = load_meta_file(root)[1]
    # _verify_archive(root, file, md5)

    val_root = os.path.join(root, folder)
    # extract_archive(os.path.join(root, file), val_root)

    images = sorted([os.path.join(val_root, image) for image in os.listdir(val_root)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(val_root, wnid))

    for wnid, img_file in zip(wnids, images):
        shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))


def load_class():
    wnid_to_class = torch.load(META_FILE)[0]
    classes =[x[0] for x in wnid_to_class.values()]

    return tuple(classes)


