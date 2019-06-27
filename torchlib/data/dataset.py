from PIL import Image
from torch.utils.data import Dataset


class Nested:

    def __init__(self, nested):
        self.len = Nested.check_len(nested)
        self.nested = nested

    def __getitem__(self, i):
        return Nested.nested_getitem(self.nested, i)

    def __len__(self):
        return self.len

    @staticmethod
    def check_len(nested):
        def flatten_lens(nested):
            if isinstance(nested, dict):
                lens = []
                for v in nested.values():
                    lens += flatten_lens(v)
            elif isinstance(nested, tuple):
                lens = []
                for v in nested:
                    lens += flatten_lens(v)
            else:
                lens = [len(nested)]
            return lens

        lens = flatten_lens(nested)
        flags = [l == lens[0] for l in lens]
        assert all(flags), 'Nested items should have the same length!'
        return lens[0]

    @staticmethod
    def nested_getitem(nested, i):
        if isinstance(nested, dict):
            nested_i = {}
            for k, v in nested.items():
                nested_i[k] = Nested.nested_getitem(v, i)
        elif isinstance(nested, tuple):
            nested_i = ()
            for v in nested:
                nested_i += (Nested.nested_getitem(v, i),)
        else:
            nested_i = nested[i]
        return nested_i


class MemoryDataDataset(Dataset):

    def __init__(self, memory_data, map_fn=None):
        """MemoryDataDataset.

        Parameters
        ----------
        memory_data : nested structure of tensors/ndarrays/lists

        """
        self.memory_data = Nested(memory_data)
        self.map_fn = map_fn

    def __len__(self):
        return len(self.memory_data)

    def __getitem__(self, i):
        item = self.memory_data[i]
        if self.map_fn:
            item = self.map_fn(item)
        return item


class DiskImageDataset(MemoryDataDataset):

    def __init__(self, img_paths, labels=None, map_fn=None):
        """MemoryDataDataset.

        Parameters
        ----------
        labels : nested structure of tensors/ndarrays/lists

        """
        # nest data
        self.img_paths = img_paths
        self.labels = labels
        if labels is None:
            memory_data = img_paths
            parse_fn = lambda path: DiskImageDataset.pil_loader(path)
        else:
            memory_data = (img_paths, labels)
            parse_fn = lambda path_label: (DiskImageDataset.pil_loader(path_label[0]), path_label[1])

        # map function
        if map_fn:  # fuse `map_fn` and `parse_fn`
            map_fn_ = lambda path_or_path_label: map_fn(parse_fn(path_or_path_label))
        else:
            map_fn_ = parse_fn

        super().__init__(memory_data, map_fn_)

    @staticmethod
    def pil_loader(path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
