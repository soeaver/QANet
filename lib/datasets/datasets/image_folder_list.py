from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, make_dataset, DatasetFolder


class ImageFolderList(DatasetFolder):
    def __init__(self, root_list, transform=None, target_transform=None, loader=default_loader):
        if not isinstance(root_list, (list, tuple)):
            raise RuntimeError("dataset_list should be a list of strings, got {}".format(dataset_list))

        super(ImageFolderList, self).__init__(
            root_list[0], loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform
        )
        if len(root_list) > 1:
            for root in root_list[1:]:
                classes, class_to_idx = self._find_classes(root)
                for k in class_to_idx.keys():
                    class_to_idx[k] += len(self.classes)
                samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
                self.classes += classes
                self.class_to_idx.update(class_to_idx)
                self.samples += samples
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples
