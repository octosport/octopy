def assert_positive_int(x, error_txt=""):
    assert isinstance(x, int) & (x >= 0), error_txt


def assert_positive(x, error_txt=""):
    assert x >= 0, error_txt
