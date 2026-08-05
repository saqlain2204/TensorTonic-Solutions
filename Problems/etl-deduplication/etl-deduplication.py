def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    if strategy == "first":
        seen = set()
        result = []

        for record in records:
            key = tuple(record[col] for col in key_columns)
            if key not in seen:
                seen.add(key)
                result.append(record)

        return result

    elif strategy == "last":
        last = {}

        for record in records:
            key = tuple(record[col] for col in key_columns)
            last[key] = record

        result = []
        added = set()

        for record in records:
            key = tuple(record[col] for col in key_columns)
            if key not in added:
                result.append(last[key])
                added.add(key)

        return result

    elif strategy == "most_complete":
        best = {}

        for record in records:
            key = tuple(record[col] for col in key_columns)
            none_count = sum(v is None for v in record.values())

            if key not in best:
                best[key] = (none_count, record)
            else:
                if none_count < best[key][0]:
                    best[key] = (none_count, record)

        result = []
        added = set()

        for record in records:
            key = tuple(record[col] for col in key_columns)
            if key not in added:
                result.append(best[key][1])
                added.add(key)

        return result

    else:
        raise ValueError("Invalid strategy")