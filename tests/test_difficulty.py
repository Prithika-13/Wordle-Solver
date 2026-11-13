from src.difficulty import word_features


def test_word_features_clusters_and_rarity():
    df = word_features().set_index("word")
    assert len(df) == 2309
    assert "catch" in df.loc["batch", "cluster_members"]
    assert df.loc["batch", "cluster_size"] >= 5          # batch/catch/hatch/latch/match/patch/watch...
    assert df.loc["boxer", "min_letter_freq"] < df.loc["crane", "min_letter_freq"]  # x is rare
    assert df.loc["poppy", "dup_letters"] == 2           # p,p,p,y -> 2 repeats
