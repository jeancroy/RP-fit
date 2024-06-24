# Similar to spreadsheet VLOOKUP()
def lookup_table(
        source_series,
        table,
        lookup_column_name,
        return_column_name
):
    return (
        # Map can use a Series (column) as key-value pairs.
        # Index is the lookup key.
        # All of these return a copy or a view, original is untouched.
        source_series.map(
            table.set_index(lookup_column_name)[return_column_name]
        )
        .rename(return_column_name)
    )
