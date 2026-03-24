def build_search_index(records: list[dict]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for row_id, record in enumerate(records):
        for token in record["title"].lower().split():
            index.setdefault(token, []).append(row_id)
    return index


def search_titles(index: dict[str, list[int]], term: str) -> list[int]:
    return index.get(term.lower(), [])
