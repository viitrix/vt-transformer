from __future__ import annotations

from typing import Any, Dict, Type

import torch


def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        serialized["dtype"] = str(self.dtype)
        # store as a plain (JSON-readable) list of numbers
        serialized["data"] = self.tolist()
        return serialized

    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        dtype_str = data["dtype"].replace("torch.", "")
        torch_dtype = getattr(torch, dtype_str)
        return torch.tensor(data["data"], dtype=torch_dtype)

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
