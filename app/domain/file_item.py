from dataclasses import dataclass


@dataclass(frozen=True)
class FileItem:
    filename: str
    content_type: str
    data: bytes
