# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy


from .efficientnet_lite import EfficientNetLite
from .ghostnet import GhostNet

from .repvgg import RepVGG

from .shufflenetv2 import ShuffleNetV2

from .esnet import ESNet


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
   
    if name == "ShuffleNetV2":
        return ShuffleNetV2(**backbone_cfg)
    elif name == "GhostNet":
        return GhostNet(**backbone_cfg)
    
    elif name == "EfficientNetLite":
        return EfficientNetLite(**backbone_cfg)
    
    elif name == "RepVGG":
        return RepVGG(**backbone_cfg)

    elif name == "esnet":
        return ESNet(**backbone_cfg)
    else:
        raise NotImplementedError
