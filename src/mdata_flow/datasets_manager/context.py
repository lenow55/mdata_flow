from enum import Enum


class DsContext(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALID = "validate"
    EMPTY = None
