"""
This module contains helper functions and utilities for teenygrad.
It defines common operations, platform detection, environment variable handling,
and data type definitions used throughout the codebase.
"""

from typing import Union, Tuple, Iterator, Optional, Final, Any
import os, functools, platform
import numpy as np
from math import prod # noqa: F401 # pylint:disable=unused-import
from dataclasses import dataclass

# Platform detection for OS-specific behavior
OSX = platform.system() == "Darwin"

# Utility functions
def dedup(x): return list(dict.fromkeys(x))   # retains list order and removes duplicates
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x  # Normalizes arguments to support both positional and unpacked tuples
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x  # Creates a tuple of length cnt from int or existing tuple
def flatten(l:Iterator): return [item for sublist in l for item in sublist]  # Flattens a 2D list/iterator into a 1D list
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__))  # Returns indices that would sort the input sequence
def all_int(t: Tuple[Any, ...]) -> bool: return all(isinstance(s, int) for s in t)  # Checks if all elements in tuple are integers
def round_up(num, amt:int): return (num+amt-1)//amt * amt  # Rounds up num to the nearest multiple of amt

# Environment variable handling with caching for performance
@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

# Debug flag controlled via environment variable
DEBUG = getenv("DEBUG")
# Continuous Integration detection
CI = os.getenv("CI", "") != ""

# Data type definition for tensor operations
@dataclass(frozen=True, order=True)
class DType:
  """
  Immutable dataclass representing a data type in teenygrad.
  
  Attributes:
    priority: Determines casting order when operations involve multiple types
    itemsize: Size in bytes of a single element
    name: String representation of the type
    np: Corresponding NumPy type (may be None for types not in NumPy)
    sz: Default element size multiplier (usually 1)
  """
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  sz: int = 1
  def __repr__(self): return f"dtypes.{self.name}"

class dtypes:
  """
  Class containing all supported data types in teenygrad.
  
  Provides static methods for type checking and conversion,
  as well as constants for all supported data types.
  """
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType)-> bool: return x in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  
  @staticmethod
  def is_float(x: DType) -> bool: return x in (dtypes.float16, dtypes.float32, dtypes.float64)
  
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  
  @staticmethod
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]  # Converts NumPy dtype to teenygrad's DType
  
  # Boolean type
  bool: Final[DType] = DType(0, 1, "bool", np.bool_)
  
  # Floating point types with different precisions
  float16: Final[DType] = DType(9, 2, "half", np.float16)
  half = float16  # Alias
  float32: Final[DType] = DType(10, 4, "float", np.float32)
  float = float32  # Alias
  float64: Final[DType] = DType(11, 8, "double", np.float64)
  double = float64  # Alias
  
  # Signed integer types with different bit widths
  int8: Final[DType] = DType(1, 1, "char", np.int8)
  int16: Final[DType] = DType(3, 2, "short", np.int16)
  int32: Final[DType] = DType(5, 4, "int", np.int32)
  int64: Final[DType] = DType(7, 8, "long", np.int64)
  
  # Unsigned integer types with different bit widths
  uint8: Final[DType] = DType(2, 1, "unsigned char", np.uint8)
  uint16: Final[DType] = DType(4, 2, "unsigned short", np.uint16)
  uint32: Final[DType] = DType(6, 4, "unsigned int", np.uint32)
  uint64: Final[DType] = DType(8, 8, "unsigned long", np.uint64)

  # Brain floating point format (not supported in NumPy)
  bfloat16: Final[DType] = DType(9, 2, "__bf16", None)

# Dictionary mapping type names to DType objects for fast lookups
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}

# Placeholder variables marked for removal
PtrDType, ImageDType, IMAGE = None, None, 0  # junk to remove
