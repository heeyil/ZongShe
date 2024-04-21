import pickle
import sys
import os
import struct
import warnings
import io
import zipfile
from typing import Any, BinaryIO, Callable, cast, Dict, Optional, Type, Tuple, Union, IO, List
from typing_extensions import TypeAlias, TypeGuard  # Python 3.10+


FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
MAP_LOCATION: TypeAlias = Optional[Union[Callable[[torch.Tensor, str], torch.Tensor], torch.device, str, Dict[str, str]]]
STORAGE: TypeAlias = Union[Storage, torch.storage.TypedStorage, torch.UntypedStorage]


def _is_path(name_or_buffer) -> TypeGuard[Union[str, os.PathLike]]:
    return isinstance(name_or_buffer, (str, os.PathLike))


class _opener:
  def __init__(self, file_like):
    self.file_like = file_like

  def __enter__(self):
    return self.file_like

  def __exit__(self, *args):
    pass


class _open_zipfile_writer_file(_opener):
  r"""创建一个用于写入 zip 文件的文件写入器"""
    def __init__(self, name) -> None:
        self.file_stream = None
        self.name = str(name)
        try:
            # 将name编码为 ASCII
            self.name.encode('ascii')
        except UnicodeEncodeError:
            # 编码不是 ASCII，文件名中包含非 ASCII 字符
            self.file_stream = io.FileIO(self.name, mode='w')
            super().__init__(self.file_stream)
        else:
            super().__init__(zipfile.ZipFile(self.name, 'w'))

    def __exit__(self, *args) -> None:
        if self.file_stream is not None:
            self.file_stream.close()


class _open_zipfile_writer_buffer(_opener):
  # 这个玩意感觉平时用球不到，不写了


def _open_zipfile_writer(name_or_buffer):
    container: Type[_opener]
    if _is_path(name_or_buffer):
        # 一般都是用这个
        container = _open_zipfile_writer_file
    else:
        container = _open_zipfile_writer_buffer
    return container(name_or_buffer)


def save(
  obj: object,
  f: FILE_LIKE
  pickle_module: Any = pickle,
  pickle_protocol: int = DEFAULT_PROTOCOL,
  _use_new_zipfile_serialization: bool = True,
  _disable_byteorder_record: bool = False
) -> None:
  
  # 下面这两个有点麻烦，先放在这不实现
  # _check_dill_version(pickle_module)
  # _check_save_filelike(f)
  
  if _use_new_zipfile_serialization:
      with _open_zipfile_writer(f) as opened_zipfile:
          _save(obj, opened_zipfile, pickle_module, pickle_protocol, _disable_byteorder_record)
          return
  else:
      with _open_file_like(f, 'wb') as opened_file:
          _legacy_save(obj, opened_file, pickle_module, pickle_protocol)


def _save(obj, zip_file, pickle_module, pickle_protocol, _disable_byteorder_record):
    serialized_storages = {}
    id_map: Dict[int, str] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: Dict = {}

    def persistent_id(obj):
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):

            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            'Cannot save multiple tensors or storages that '
                            'view the same data as different types')
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            location = location_tag(storage)
            serialized_storages[storage_key] = storage

            return ('storage',
                    storage_type,
                    storage_key,
                    location,
                    storage_numel)

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    zip_file.write_record('data.pkl', data_value, len(data_value))

    # Write byte order marker
    if not _disable_byteorder_record:
        if sys.byteorder not in ['little', 'big']:
            raise ValueError('Unknown endianness type: ' + sys.byteorder)

        zip_file.write_record('byteorder', sys.byteorder, len(sys.byteorder))

    # Write each tensor to a file named tensor/the_tensor_key in the zip archive
    for key in sorted(serialized_storages.keys()):
        name = f'data/{key}'
        storage = serialized_storages[key]
        # given that we copy things around anyway, we might use storage.cpu()
        # this means to that to get tensors serialized, you need to implement
        # .cpu() on the underlying Storage
        if storage.device.type != 'cpu':
            storage = storage.cpu()
        # Now that it is on the CPU we can directly copy it into the zip file
        num_bytes = storage.nbytes()
        zip_file.write_record(name, storage, num_bytes)
