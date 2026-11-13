"""Static contact geometry, with explicit units, pixel mapping and alignment gates.

The installed SDK candidate is read as bytes, never imported or dlopened. Unknown
binary revisions fail closed. No device, network or SDK lifecycle calls occur here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .data import FINGERS, force_to_world

SDK_LIBRARY = Path("/opt/sharpa-wave-sdk/lib/libsharpa-wave-sdk.so")
SDK_SHA256 = "841951806d74dc2adbcbe443f4a21d8668fb2a19ba315b5af36708a26ae7e5dc"
CHECKS = ("pixel_mapping", "units", "alignment", "normal_convention", "finger_model")


def rigid_transform(value: Any) -> np.ndarray:
    """Validate a finite rigid S-to-E or E-to-W transform, including its last row."""
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Expected finite 4x4 transform")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("Invalid rigid transform last row")
    force_to_world(transform[:3, :3], [0, 0, 0])
    return transform.copy()


def transform_contact(transform: Any, point: Any, normal: Any, force: Any) -> tuple[np.ndarray, ...]:
    """Translate points only; rotate normals/forces and preserve force magnitude."""
    transform = rigid_transform(transform)
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Invalid contact point")
    normal = force_to_world(transform[:3, :3], normal)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("Zero normal")
    return (transform[:3, :3] @ point + transform[:3, 3], normal / norm, force_to_world(transform[:3, :3], force))


@dataclass(frozen=True)
class ContactGeometry:
    """Coordinates at one selected pixel; raw units and verification stay attributable."""

    row: int
    col: int
    p_S: np.ndarray
    n_S: np.ndarray
    p_E: np.ndarray
    n_E: np.ndarray
    verification: str
    source: str
    units: str


class GeometryMap:
    """One immutable geometry profile, shared by a documented set of fingers."""

    def __init__(
        self,
        xyz: Any,
        normals: Any,
        valid_mask: Any,
        *,
        meters_per_unit: float,
        T_E_S: Any,
        uv_to_col_row: Any,
        verified: bool,
        evidence: dict[str, str],
        source: str,
        units: str,
    ) -> None:
        if np.asarray(xyz).dtype.kind not in "fiu" or np.asarray(normals).dtype.kind not in "fiu":
            raise ValueError("XYZ and normals require real numeric arrays")
        if np.asarray(valid_mask).dtype.kind != "b":
            raise ValueError("Geometry valid mask must be boolean")
        self.xyz, self.normals = np.array(xyz, dtype=float, copy=True), np.array(normals, dtype=float, copy=True)
        self.mask = np.array(valid_mask, dtype=bool, copy=True)
        if self.xyz.ndim != 3 or self.xyz.shape[-1] != 3 or min(self.xyz.shape[:2]) < 1:
            raise ValueError("XYZ must be H x W x 3")
        if self.normals.shape != self.xyz.shape or self.mask.shape != self.xyz.shape[:2]:
            raise ValueError("Geometry/normal/mask shapes disagree")
        if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
            raise ValueError("Explicit positive meters_per_unit required")
        unit = units.removeprefix("candidate ").split(" ", 1)[0] if isinstance(units, str) else ""
        scales = {"m": 1.0, "mm": 0.001}
        if unit not in scales or not np.isclose(meters_per_unit, scales[unit], rtol=0, atol=1e-12):
            raise ValueError("Geometry units must declare m or mm with matching meters_per_unit")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Explicit geometry source required")
        if not isinstance(evidence, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in evidence.items()
        ):
            raise ValueError("Geometry evidence requires string key/value pairs")
        if type(verified) is not bool or (verified and any(not evidence.get(k, "").strip() for k in CHECKS)):
            raise ValueError("Verified geometry requires evidence for every verification gate")
        self.T_E_S = rigid_transform(T_E_S)
        self.uv_to_col_row = np.asarray(uv_to_col_row, dtype=float).copy()
        if self.uv_to_col_row.shape != (2, 3) or not np.isfinite(self.uv_to_col_row).all():
            raise ValueError("Explicit finite 2x3 UV-to-col/row mapping required")
        if abs(np.linalg.det(self.uv_to_col_row[:, :2])) < 1e-12:
            raise ValueError("Singular pixel mapping")
        self.meters_per_unit, self.verified = float(meters_per_unit), verified
        self.evidence, self.source, self.units = dict(evidence), source, units
        # Invalid finite/normal samples remain holes, never silently filled.
        self.mask &= np.isfinite(self.xyz).all(axis=2) & np.isfinite(self.normals).all(axis=2)
        self.mask &= np.linalg.norm(self.normals, axis=2) > 1e-12
        for array in (self.xyz, self.normals, self.mask, self.T_E_S, self.uv_to_col_row):
            array.flags.writeable = False

    def lookup(self, uv: Any, allow_unverified: bool = False) -> ContactGeometry:
        """Round nearest pixel (half upward), reject boundaries/holes, never clamp.

        Raises:
            ValueError: If alignment is unverified without opt-in, or the selected pixel is invalid.
        """
        if not self.verified and not allow_unverified:
            raise ValueError("UNVERIFIED GEOMETRY: diagnostic display is disabled")
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError("Invalid UV")
        col_f, row_f = self.uv_to_col_row @ np.r_[uv, 1]
        height, width = self.mask.shape
        if not 0 <= col_f <= width - 1 or not 0 <= row_f <= height - 1:
            raise ValueError(f"UV outside lookup pixel-center domain [0,{width - 1}] x [0,{height - 1}]")
        row, col = int(np.floor(row_f + 0.5)), int(np.floor(col_f + 0.5))
        if not self.mask[row, col]:
            raise ValueError("Nearest pixel is an invalid geometry hole; no snapping to another pixel")
        p_s, n_s = self.xyz[row, col].copy(), self.normals[row, col].copy()
        n_s /= np.linalg.norm(n_s)
        p_e, n_e, _ = transform_contact(self.T_E_S, p_s * self.meters_per_unit, n_s, [0, 0, 0])
        status = "VERIFIED BY CONFIG EVIDENCE" if self.verified else "UNVERIFIED DIAGNOSTIC"
        return ContactGeometry(row, col, p_s, n_s, p_e, n_e, status, self.source, self.units)


def sdk_hb1_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read named embedded float32 maps from the audited SDK build's ELF sections.

    Sizes, layout and symbol names are pinned by the full library hash. This is
    local asset access, not a public SDK API; other builds must be audited first.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SDK_SHA256:
        raise ValueError("SDK binary hash differs from audited 5.0.10.6; static maps unavailable")
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected little-endian ELF64")
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    section_size, section_count = struct.unpack_from("<HH", data, 58)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, section_offset + i * section_size) for i in range(section_count)
    ]
    wanted = {
        f"tactile_map_{kind}_hb1_{field}_{edge}"
        for kind in ("general", "thumb")
        for field in ("point", "normal")
        for edge in ("begin", "end")
    }
    symbols = {}
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB, with its associated string table.
            continue
        strings = sections[section[6]]
        names = data[strings[4] : strings[4] + strings[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name_offset, _, _, index, value, _ = struct.unpack_from("<IBBHQQ", data, offset)
            name = names[name_offset : names.index(b"\0", name_offset)].decode("utf-8", errors="replace")
            if name in wanted:
                symbols[name] = (index, value)
    if symbols.keys() != wanted:
        raise ValueError("Embedded HB1 map symbols are missing")
    result = {}
    for kind in ("general", "thumb"):
        for field in ("point", "normal"):
            prefix = f"tactile_map_{kind}_hb1_{field}"
            index, start = symbols[prefix + "_begin"]
            end_index, end = symbols[prefix + "_end"]
            section = sections[index]
            if index != end_index or end - start != 240 * 240 * 3 * 4:
                raise ValueError("Unexpected static map byte extent")
            offset = section[4] + start - section[3]
            result[f"{kind}_{field}"] = np.frombuffer(data, dtype="<f4", count=240 * 240 * 3, offset=offset).reshape(
                240, 240, 3
            )
    return result


class GeometryCatalog:
    """Per-finger maps and independent failures; one missing profile cannot stop the viewer."""

    def __init__(
        self,
        maps: dict[str, GeometryMap] | None = None,
        errors: dict[str, str] | None = None,
        profile_names: dict[str, str] | None = None,
    ) -> None:
        self.maps, self.errors = maps or {}, errors or {}
        self.profile_names = dict(profile_names or {})

    def lookup(self, finger: str, uv: Any, allow_unverified: bool) -> ContactGeometry:
        """Look up only the explicitly mapped finger profile."""
        if finger not in self.maps:
            raise ValueError("GEOMETRY UNAVAILABLE: " + self.errors.get(finger, "no finger profile"))
        return self.maps[finger].lookup(uv, allow_unverified)

    @classmethod
    def installed_candidate(cls, library: Path = SDK_LIBRARY) -> GeometryCatalog:
        """Load audited HB1 bytes with explicit, unverified mm/identity/UV candidates."""
        try:
            arrays = sdk_hb1_arrays(library)
            profiles = {}
            for kind in ("general", "thumb"):
                xyz, normal = arrays[f"{kind}_point"], arrays[f"{kind}_normal"]
                profiles[kind] = GeometryMap(
                    xyz,
                    normal,
                    np.linalg.norm(normal, axis=2) > 0.5,
                    meters_per_unit=0.001,
                    T_E_S=np.eye(4),
                    uv_to_col_row=[[1, 0, 0], [0, 1, 0]],
                    verified=False,
                    evidence={},
                    units="candidate mm (not certified)",
                    source=f"SDK 5.0.10.6 HB1 {kind}; {library}; sha256={SDK_SHA256}",
                )
            names = {f: "thumb" if f == "thumb" else "general" for f in FINGERS}
            return cls({f: profiles[n] for f, n in names.items()}, profile_names=names)
        except (OSError, ValueError, struct.error) as exc:
            return cls(errors={f: str(exc) for f in FINGERS})

    @classmethod
    def from_config(cls, path: Path) -> GeometryCatalog:
        """Read explicit external NPY profiles and transforms; never infer identity or units."""
        maps, errors, names = {}, {}, {}
        try:
            config = json.loads(path.read_text())
            if config["schema_version"] != 1:
                raise ValueError("Unsupported geometry config schema")
            profiles = config["profiles"]
            loaded: dict[str, GeometryMap] = {}
            for finger in FINGERS:
                try:
                    name = config["fingers"][finger]
                    if name not in loaded:
                        item = dict(profiles[name])
                        arrays = {
                            key: np.load(path.parent / item.pop(key), allow_pickle=False)
                            for key in ("xyz", "normals", "valid_mask")
                        }
                        loaded[name] = GeometryMap(**arrays, **item)
                    maps[finger] = loaded[name]
                    names[finger] = name
                except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                    errors[finger] = str(exc)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors = {f: str(exc) for f in FINGERS}
        return cls(maps, errors, names)
