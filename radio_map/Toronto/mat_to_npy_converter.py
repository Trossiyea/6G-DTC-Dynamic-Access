#!/usr/bin/env python3
"""
MAT文件导出为NumPy .npy

特性：
1) 支持 MATLAB v7.3 (HDF5) 与传统 .mat（scipy.io.loadmat）
2) 自动选择数值数据键（优先 XdB_recon_tensor，否则选择最大数值数组）
3) 尝试从转换信息中推断单位（例如 dBW / dBm）
4) 生成同名 _meta.json 记录键名/形状/单位/范围等
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import h5py
from scipy.io import loadmat


@dataclass(frozen=True)
class MatLoadResult:
    format_type: str  # 'v7.3' | 'legacy'
    data: Dict[str, Any]


def _load_mat(filepath: str) -> MatLoadResult:
    # MATLAB v7.3 文件是 HDF5（通常带 userblock 头部），优先用 h5py 尝试。
    try:
        data: Dict[str, Any] = {}
        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    data[key] = np.array(obj)
                else:
                    data[key] = obj
        return MatLoadResult(format_type="v7.3", data=data)
    except Exception:
        pass

    mat_data = loadmat(filepath)
    user_vars = {k: v for k, v in mat_data.items() if not k.startswith("__")}
    return MatLoadResult(format_type="legacy", data=user_vars)


def _numeric_arrays(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in ["f", "i"]:
            out[key] = value
    return out


def _pick_data_key(numeric: Dict[str, np.ndarray], preferred_key: Optional[str]) -> str:
    if preferred_key:
        if preferred_key not in numeric:
            raise KeyError(f"未找到数值键: {preferred_key}")
        return preferred_key
    if "XdB_recon_tensor" in numeric:
        return "XdB_recon_tensor"
    if not numeric:
        raise ValueError("未找到可导出的数值数组")
    return max(numeric.keys(), key=lambda k: numeric[k].size)


def _decode_h5_str(arr: Any) -> Optional[str]:
    if arr is None:
        return None
    if isinstance(arr, bytes):
        return arr.decode("utf-8", errors="ignore")
    if isinstance(arr, np.ndarray) and arr.dtype.kind == "S":
        try:
            return b"".join(arr.flatten()).decode("utf-8", errors="ignore")
        except Exception:
            return None
    return None


def _infer_units_from_v73(filepath: str, data_key: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    返回： (units, extra_meta)
    仅对 v7.3 (HDF5) 尝试读取 conversion_* 信息。
    """
    extra: Dict[str, str] = {}
    try:
        with h5py.File(filepath, "r") as f:
            def get_str(name: str) -> Optional[str]:
                ds = f.get(name)
                if ds is None:
                    return None
                try:
                    return _decode_h5_str(ds[()])
                except Exception:
                    return None

            converted_key = get_str("conversion_converted_key")
            converted_units = get_str("conversion_converted_units")
            original_units = get_str("conversion_original_units")
            formula = get_str("conversion_conversion_formula")
            ts = get_str("conversion_conversion_timestamp")

        if converted_key:
            extra["conversion_converted_key"] = converted_key
        if converted_units:
            extra["conversion_converted_units"] = converted_units
        if original_units:
            extra["conversion_original_units"] = original_units
        if formula:
            extra["conversion_conversion_formula"] = formula
        if ts:
            extra["conversion_conversion_timestamp"] = ts

        if converted_key == data_key and converted_units:
            return converted_units, extra
    except Exception:
        pass
    return None, extra


def _infer_units_from_filename(filepath: str) -> Optional[str]:
    base = os.path.basename(filepath).lower()
    if "_dbm" in base:
        return "dBm"
    if "_dbw" in base:
        return "dBW"
    return None


def _append_units_to_basename(path_no_ext: str, units: Optional[str]) -> str:
    if not units:
        return path_no_ext
    basename = os.path.basename(path_no_ext)
    dirname = os.path.dirname(path_no_ext)
    lower = basename.lower()
    suffix = f"_{units}".lower()
    if lower.endswith(suffix):
        return path_no_ext
    return os.path.join(dirname, f"{basename}_{units}")


def convert_one(
    input_mat: str,
    output_npy: Optional[str],
    data_key: Optional[str],
    units_override: Optional[str],
    write_meta: bool,
) -> Tuple[str, str]:
    if not os.path.exists(input_mat):
        raise FileNotFoundError(f"输入文件不存在: {input_mat}")

    loaded = _load_mat(input_mat)
    numeric = _numeric_arrays(loaded.data)
    picked_key = _pick_data_key(numeric, data_key)
    arr = numeric[picked_key]

    units: Optional[str] = units_override
    extra_meta: Dict[str, str] = {}
    if units is None and loaded.format_type == "v7.3":
        units, extra_meta = _infer_units_from_v73(input_mat, picked_key)
    if units is None:
        units = _infer_units_from_filename(input_mat)

    if output_npy is None:
        base_no_ext, _ = os.path.splitext(input_mat)
        base_no_ext = _append_units_to_basename(base_no_ext, units)
        output_npy = f"{base_no_ext}.npy"

    output_npy = os.path.abspath(output_npy)
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)

    np.save(output_npy, np.ascontiguousarray(arr))

    meta_path = os.path.splitext(output_npy)[0] + "_meta.json"
    if write_meta:
        meta: Dict[str, Any] = {
            "source_mat": os.path.abspath(input_mat),
            "mat_format": loaded.format_type,
            "data_key": picked_key,
            "units": units,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "nan_count": int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0,
        }
        meta.update(extra_meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return output_npy, meta_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MATLAB .mat to NumPy .npy (with optional _meta.json).")
    parser.add_argument("input_mat", nargs="+", help="输入 .mat 文件路径（可多个）")
    parser.add_argument("-k", "--key", default=None, help="指定导出的变量键名（默认自动选择）")
    parser.add_argument("-o", "--output", default=None, help="输出 .npy 路径（仅当输入为单文件时可用）")
    parser.add_argument("--units", default=None, help="强制指定单位（例如 dBW/dBm）")
    parser.add_argument("--no-meta", action="store_true", help="不生成 _meta.json")
    args = parser.parse_args()

    if args.output and len(args.input_mat) != 1:
        parser.error("当指定 -o/--output 时，只能输入一个 .mat 文件")

    for p in args.input_mat:
        out_npy, out_meta = convert_one(
            input_mat=p,
            output_npy=args.output,
            data_key=args.key,
            units_override=args.units,
            write_meta=(not args.no_meta),
        )
        if args.no_meta:
            print(out_npy)
        else:
            print(out_npy)
            print(out_meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
