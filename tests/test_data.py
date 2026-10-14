import pandas as pd

from app.data import guess_id, guess_target, load_table, profile_frame


def test_semicolon_cp1251_decimal_comma(tmp_path):
    path = tmp_path / "bank.csv"
    path.write_bytes("id;город;сумма\n1;Астана;1,5\n2;Алматы;2,25\n3;Шымкент;10\n".encode("cp1251"))
    df = load_table(path)
    assert list(df.columns) == ["id", "город", "сумма"]
    assert df["город"].tolist() == ["Астана", "Алматы", "Шымкент"]
    assert df["сумма"].tolist() == [1.5, 2.25, 10.0]


def test_excel_and_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.to_excel(tmp_path / "f.xlsx", index=False)
    df.to_parquet(tmp_path / "f.parquet")
    assert load_table(tmp_path / "f.xlsx").equals(load_table(tmp_path / "f.parquet"))


def test_roles_and_guesses(bank):
    train, test = bank
    roles = profile_frame(train, "default").set_index("name")["role"]
    assert roles["client_id"] == "id"
    assert roles["open_date"] == "datetime"
    assert roles["city"] == "categorical"
    assert roles["default"] == "target"
    assert guess_target(train, test) == "default"
    assert guess_id(train, test) == "client_id"
