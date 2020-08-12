import torch
import torch.nn
import itertools

import torch.functional as F
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Set, Optional

@dataclass
class SupernetBlockSpec:
    """Supernet block specification
    """
    kind: str
    params: dict

    @classmethod
    def from_dict(cls, spec: Dict) -> 'SupernetBlockSpec':
        params = {k: v for k, v in spec.items() if k != 'kind'}
        return cls(kind=spec['kind'], params=params)

@dataclass
class SupernetLayerSpec:
    """Supernet layer specification
    """
    out_channels: int
    blocks: List[SupernetBlockSpec]

    @classmethod
    def from_dict(cls, spec: Dict) -> 'SupernetLayerSpec':
        return cls(
            out_channels=spec['out_channels'],
            blocks={
                name: SupernetBlockSpec.from_dict(params) 
                for name, params in spec['blocks'].items()
            }
        )

@dataclass
class SupernetClassifierSpec:
    """Supernet model config
    """
    input_size: Tuple[int, int]
    num_input_channels: int
    num_output_layer_channels: int
    num_classes: int
    supernet_layers: List[SupernetLayerSpec]

    @classmethod
    def from_dict(cls, spec: Dict) -> 'SupernetClassifierSpec':
        return cls(
            input_size=spec['input_size'],
            num_input_channels=spec['num_input_channels'],
            num_output_layer_channels=spec['num_output_layer_channels'],
            num_classes=spec['num_classes'],
            supernet_layers=[SupernetLayerSpec.from_dict(x) for x in spec['supernet_layers']]
        )


def get_output_size(module: torch.nn.Module, input_size: torch.Size, **params) -> torch.Size:
    """ Compute output size for input of given size
        For Conv2d this can be computed analytically, but from maintanence and generalization standpoint I much prefer brute-force method
    """
    in_tensor = torch.zeros(input_size)
    out_tensor = module(in_tensor, **params)
    return out_tensor.size()


def build_block(block_spec: SupernetBlockSpec, in_channels: int, out_channels: int) -> torch.nn.Module:
    """Build a block according to specification.
       This is a simple extension point for future experiments, albeit not necessary for given task
    """

    if block_spec.kind == 'conv2d':
        return torch.nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               **block_spec.params)

    if block_spec.kind == 'identity':
        return torch.nn.Identity()
    
    raise ValueError(f'Unknown block kind: {block_spec.kind}') 


class PadToWidth(torch.nn.Module):

    def __init__(self, embedding_size: int, fill_value: float = 0):
        super(PadToWidth, self).__init__()
        self._embedding_size = embedding_size
        self._fill_value = fill_value
    
    def forward(self, x):
        input_size = x.size()
        target_size = torch.Size((*input_size[:-1], self._embedding_size))
        result = torch.full(target_size, dtype=x.dtype, fill_value=self._fill_value)
        result[..., :input_size[-1]] = x
        return result


class SupernetLayer(torch.nn.Module):

    def __init__(self,
                 blocks: Dict[str, torch.nn.Module],
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 pooling: torch.nn.Module = torch.nn.MaxPool2d(kernel_size=2),
                 default_block: Optional[str] = None):

        """Supernet cell.
           Implements a NN layer with switchable inner block
           Active block can be selected by calling .select_block

        Args:
            blocks (Dict[str, torch.nn.Module]): inner blocks. 
                                                 Only one is active at any time, although this behaiviour could be changed by paddnig and summing the outputs
            activation (torch.nn.Module, optional): Activation function. Defaults to torch.nn.ReLU.
            pooling (torch.nn.Module, optional): Pooling operation. Defaults to torch.nn.MaxPool2d with 2x2 kernel
            default_block (Optional[str], optional): Name of the default block. Defaults to None

        Raises:
            ValueError: [description]
        """

        if len(blocks) == 0:
            raise ValueError("At least one block should be passed")

        super(SupernetLayer, self).__init__()

        self._blocks = torch.nn.ModuleDict(blocks) 
        self._active_block = None
        self._activation = activation
        self._pooling = pooling

        self.select_block(default_block)

    @classmethod
    def from_spec(cls, spec: SupernetLayerSpec, in_channels: int) -> 'SupernetLayer':
        return cls(
            blocks={
                name: build_block(block_spec, in_channels, spec.out_channels)
                for name, block_spec in spec.blocks.items()
            }
        )

    @property
    def available_blocks(self) -> Set[str]:
        """Set of available layer block names
        """
        return set(self._blocks.keys())

    def get_output_sizes(self, input_size: torch.Size) -> Dict[str, torch.Size]:
        """Get dictionary with output sizes for given input size
        """
        return {
            name: get_output_size(self, input_size, with_block=name)
            for name in self.available_blocks
        }

    def select_block(self, block_name: Optional[str]):
        """Select a block with given name as active
        """

        if block_name is None:
            self._active_block = None
        elif block_name in self.available_blocks:
            self._active_block = block_name
        else:
            raise ValueError(f'Block with name {block_name} does not exist')

    def distill(self):
        """Extract torch.nn.Sequential module equivalent to current layer configuration
        """
        return torch.nn.Sequential(
            self._blocks[self._active_block],
            self._activation,
            self._pooling
        )

    def forward(self, x: torch.Tensor, with_block: Optional[str] = None) -> torch.Tensor:
        
        if self._active_block is None and with_block is None:
            raise ValueError("One of the block should be selected as active")

        selected_block = self._blocks[self._active_block] if with_block is None else self._blocks[with_block]
        x = selected_block.forward(x)
        x = self._activation(x)
        x = self._pooling(x)

        return x


class SupernetClassifier(torch.nn.Module):

    def __init__(self, 
                 input_size: Tuple[int, int],
                 num_input_channels: int, 
                 num_output_layer_channels: int,
                 num_classes: int,
                 supernet_layers: List[SupernetLayer]):

        """Supernet classifier model

        Args:
            input_size (Tuple[int, int]): input resulotion
            num_input_channels (int): number of channels in input
            num_classes (int): number of classes
            supernet_layers (List[SupernetLayer]): list of supernet layers
        """
        super(SupernetClassifier, self).__init__()

        input_size = torch.Size((1, num_input_channels, *input_size))
        possible_output_sizes = {input_size}
        for layer in supernet_layers:
            possible_output_sizes = set().union(*(
                set(layer.get_output_sizes(s).values())
                for s in possible_output_sizes
            ))
      
        self._input_size = input_size
        self._supernet_layers = torch.nn.ModuleList(supernet_layers)
        self._embedding_size = max(s.numel() for s in possible_output_sizes)
        self._output_layers = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            PadToWidth(self._embedding_size),
            torch.nn.Linear(in_features=self._embedding_size, out_features=num_output_layer_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=num_output_layer_channels, out_features=num_classes),
            torch.nn.Softmax()
        )

    @classmethod
    def from_spec(cls, spec: SupernetClassifierSpec) -> 'SupernetClassifier':

        supernet_layers = []
        in_channels = spec.num_input_channels

        for layer_spec in spec.supernet_layers:
            layer = SupernetLayer.from_spec(layer_spec, in_channels)
            in_channels = layer_spec.out_channels
            supernet_layers.append(layer)

        return cls(
            input_size=spec.input_size,
            num_input_channels=spec.num_input_channels,
            num_output_layer_channels=spec.num_output_layer_channels,
            num_classes=spec.num_classes,
            supernet_layers=supernet_layers
        )

    def get_available_configurations(self) -> Iterable[Tuple]:

        available_blocks = [
            layer.available_blocks
            for layer in self._supernet_layers
        ]

        available_configurations = itertools.product(*available_blocks)

        for configuration in available_configurations:
            yield tuple(configuration)

    def reconfigure(self, block_names: Iterable[str]):

        if len(block_names) != len(self._supernet_layers):
            raise ValueError(f'Model contains {len(self._supernet_layers)} supernet layers, but passed {len(block_names)} block names')

        for i, name in enumerate(block_names):

            layer = self._supernet_layers[i]
            if name not in layer.available_blocks:
                raise ValueError(f'Layer does not contain block {name}. Available blocks: {layer.available_block}')

            layer.select_block(name)

    def distill(self):

        supernet_layers = [layer.distill() for layer in self._supernet_layers]
        return torch.nn.Sequential(
            *supernet_layers,
            self._output_layers
        )

    def forward(self, x: torch.Tensor):

        for layer in self._supernet_layers:
            x = layer(x)

        return self._output_layers(x)
