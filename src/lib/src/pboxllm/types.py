# -*- coding: UTF-8 -*-
from typing import TypedDict


class FewShotExample(TypedDict):
    features_text: str
    label: int
