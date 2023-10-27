import evaluate
import pprint
import argparse

class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order."""
        config_str = "Configurations\n"
        config_str += pprint.pformat(self.__dict__)
        return config_str


