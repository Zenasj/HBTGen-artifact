def _create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> List[ReadItem]:
    if not isinstance(md, BytesStorageMetadata):
        try:
            local_chunks = _create_chunk_list(obj)
        except ValueError as ex:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, "
                + f"expected BytesStorageMetadata but found {type(md)}",
            ) from ex