from __future__ import annotations

from typing import Any


class PointLights:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class PerspectiveCameras:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Materials:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class SoftPhongShader:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class RasterizationSettings:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class MeshRendererWithFragments:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]: ...


class MeshRasterizer:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def cuda(self) -> MeshRasterizer: ...


class TexturesVertex:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
