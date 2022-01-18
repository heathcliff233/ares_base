from typing import Optional, List, Union, Any, Tuple

import inspect
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import numpy as np
from torch import Tensor
from torch_sparse import SparseTensor, cat

from torch_geometric.data import Data, HeteroData
#from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from torch_geometric.data.dataset import IndexType
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage, NodeStorage



def collate(
    cls,
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> Tuple[BaseData, Mapping, Mapping]:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    
    print(data_list[0].__class__)
    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # Dynamic inheritance.
    else:
        out = cls()
    
    #out = cls()
    print("before create storage")

    # Create empty stores:
    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_store_list = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device = None
    slice_dict, inc_dict = defaultdict(dict), defaultdict(dict)
    for out_store in out.stores:
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == 'num_nodes':
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == 'ptr':
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(attr, values, data_list, stores,
                                           increment)

            device = value.device if isinstance(value, Tensor) else device

            out_store[attr] = value
            if key is not None:
                slice_dict[key][attr] = slices
                inc_dict[key][attr] = incs
            else:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if (attr in follow_batch and isinstance(slices, Tensor)
                    and slices.dim() == 1):
                repeats = slices[1:] - slices[:-1]
                batch = repeat_interleave(repeats.tolist(), device=device)
                out_store[f'{attr}_batch'] = batch

        # In case the storage holds node, we add a top-level batch vector it:
        if (add_batch and isinstance(stores[0], NodeStorage)
                and stores[0].can_infer_num_nodes):
            repeats = [store.num_nodes for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(torch.tensor(repeats, device=device))

    return out, slice_dict, inc_dict


def _collate(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
) -> Tuple[Any, Any, Any]:

    elem = values[0]

    if isinstance(elem, Tensor):
        # Concatenate a list of `torch.Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        if cat_dim is None or elem.dim() == 0:
            values = [value.unsqueeze(0) for value in values]
        slices = cumsum([value.size(cat_dim or 0) for value in values])
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if incs.dim() > 1 or int(incs[-1]) != 0:
                values = [
                    value + inc.to(value.device)
                    for value, inc in zip(values, incs)
                ]
        else:
            incs = None

        if torch.utils.data.get_worker_info() is not None:
            # Write directly into shared memory to avoid an extra copy:
            numel = sum(value.numel() for value in values)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        else:
            out = None

        value = torch.cat(values, dim=cat_dim or 0, out=out)
        return value, slices, incs

    elif isinstance(elem, SparseTensor) and increment:
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        repeats = [[value.size(dim) for dim in cat_dims] for value in values]
        slices = cumsum(repeats)
        value = cat(values, dim=cat_dim)
        return value, slices, None

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `torch.Tensor`.
        value = torch.tensor(values)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if int(incs[-1]) != 0:
                value.add_(incs)
        else:
            incs = None
        slices = torch.arange(len(values) + 1)
        return value, slices, incs

    elif isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for key in elem.keys():
            value_dict[key], slice_dict[key], inc_dict[key] = _collate(
                key, [v[key] for v in values], data_list, stores, increment)
        return value_dict, slice_dict, inc_dict

    elif (isinstance(elem, Sequence) and not isinstance(elem, str)
          and isinstance(elem[0], (Tensor, SparseTensor))):
        # Recursively collate elements of lists.
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(key, [v[i] for v in values],
                                           data_list, stores, increment)
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    else:
        # Other-wise, just return the list of values as it is.
        slices = torch.arange(len(values) + 1)
        return values, slices, None


###############################################################################


def repeat_interleave(
    repeats: List[int],
    device: Optional[torch.device] = None,
) -> Tensor:
    outs = [torch.full((n, ), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


def cumsum(value: Union[Tensor, List[int]]) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.tensor(value)
    out = value.new_empty((value.size(0) + 1, ) + value.size()[1:])
    out[0] = 0
    torch.cumsum(value, 0, out=out[1:])
    return out


def get_incs(key, values: List[Any], data_list: List[BaseData],
             stores: List[BaseStorage]) -> Tensor:
    repeats = [
        data.__inc__(key, value, store)
        for value, data, store in zip(values, data_list, stores)
    ]
    if isinstance(repeats[0], Tensor):
        repeats = torch.stack(repeats, dim=0)
    else:
        repeats = torch.tensor(repeats)
    return cumsum(repeats[:-1])

class DynamicInheritance(type):
    # A meta class that sets the base class of a `Batch` object, e.g.:
    # * `Batch(Data)` in case `Data` objects are batched together
    # * `Batch(HeteroData)` in case `HeteroData` objects are batched together
    def __call__(cls, *args, **kwargs):
        base_cls = kwargs.pop('_base_cls', Data)
        print(base_cls)

        if issubclass(base_cls, Batch):
            new_cls = base_cls
        else:
            name = f'{base_cls.__name__}{cls.__name__}'
            if name not in globals():
                globals()[name] = type(name, (cls, base_cls), {})
            new_cls = globals()[name]

        params = list(inspect.signature(base_cls.__init__).parameters.items())
        for i, (k, v) in enumerate(params[1:]):
            if k == 'args' or k == 'kwargs':
                continue
            if i < len(args) or k in kwargs:
                continue
            if v.default is not inspect.Parameter.empty:
                continue
            kwargs[k] = None

        return super(DynamicInheritance, new_cls).__call__(*args, **kwargs)


class DynamicInheritanceGetter(object):
    def __call__(self, cls, base_cls):
        return cls(_base_cls=base_cls)


class Batch(metaclass=DynamicInheritance):
    r"""A data object describing a batch of graphs as one big (disconnected)
    graph.
    Inherits from :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData`.
    In addition, single graphs can be identified via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    @classmethod
    def from_data_list(cls, data_list: Union[List[Data], List[HeteroData]],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None):
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        Python list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch


    def get_example(self, idx: int) -> Union[Data, HeteroData]:
        r"""Gets the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object at index :obj:`idx`.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object."""

        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                ("Cannot reconstruct 'Data' object from 'Batch' because "
                 "'Batch' was not created via 'Batch.from_data_list()'"))

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

        return data


    def index_select(self, idx: IndexType) -> Union[List[Data], List[HeteroData]]:
        r"""Creates a subset of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects from specified
        indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects."""
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            idx = idx.flatten().tolist()

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            idx = idx.flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self.get_example(i) for i in idx]


    def __getitem__(self, idx: Union[int, np.integer, str, IndexType]) -> Any:
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            return self.get_example(idx)
        elif isinstance(idx, str) or (isinstance(idx, tuple)
                                      and isinstance(idx[0], str)):
            # Accessing attributes or node/edge types:
            return super().__getitem__(idx)
        else:
            return self.index_select(idx)

    def to_data_list(self) -> Union[List[Data], List[HeteroData]]:
        r"""Reconstructs the list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects from the
        :class:`~torch_geometric.data.Batch` object.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial objects."""
        return [self.get_example(i) for i in range(self.num_graphs)]


    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if hasattr(self, '_num_graphs'):
            return self._num_graphs
        elif hasattr(self, 'ptr'):
            return self.ptr.numel() - 1
        elif hasattr(self, 'batch'):
            return int(self.batch.max()) + 1
        else:
            raise ValueError("Can not infer the number of graphs")

    def __reduce__(self):
        state = self.__dict__.copy()
        return DynamicInheritanceGetter(), self.__class__.__bases__, state

