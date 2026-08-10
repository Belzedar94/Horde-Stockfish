# Horde material corpus

`horde_insufficient_material.csv` is copied without modification from
lichess-org/scalachess commit
`d5d47c16f65a005ca68e19bab702b02f66dd888c`:

`test-kit/src/test/resources/horde_insufficient_material.csv`

- Rows: 21,996
- Bytes: 1,061,395
- SHA-256: `1F01B4FD3AB6066EFE5A96A2AE4E0DF5074FB99377D7CAAA2ACBA04D453D53CC`

The corpus is used only to validate the color-specific
`side_has_insufficient_winning_material` rules contract. A matching row does
not, by itself, make a position an automatic draw.
