import pytest
import torch
import itertools
from .supernet import SupernetLayer, SupernetClassifier, SupernetClassifierSpec


def test_supernet_layer_forward():

    layer = SupernetLayer({
        'conv3x3': torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3)),
        'conv5x5': torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5, 5)),
        'conv7x7': torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(7, 7))
    })

    test_data = torch.rand(1, 4, 8, 8)

    with pytest.raises(ValueError):
        results = layer(test_data)

    layer.select_block('conv3x3')
    results3x3 = layer(test_data)

    layer.select_block('conv5x5')
    results5x5 = layer(test_data)

    layer.select_block('conv7x7')
    results7x7 = layer(test_data)

    assert results3x3.size != results5x5.size or not torch.allclose(results3x3, results5x5)
    assert results3x3.size != results7x7.size or not torch.allclose(results3x3, results7x7) 
    assert results5x5.size != results7x7.size or not torch.allclose(results5x5, results7x7)

def test_supernet_layer_distill():

    layer = SupernetLayer({
        'conv3x3': torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3)),
        'conv5x5': torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5, 5)),
        'conv7x7': torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(7, 7))
    })

    test_data = torch.rand(1, 4, 8, 8)

    layer.select_block('conv5x5')
    distilled_layer = layer.distill()

    assert torch.allclose(layer(test_data), distilled_layer(test_data))

@pytest.mark.parametrize('batch_size, in_channels, out_channels, width, height', [
    (6, 8, 16, 24, 32),
    (1, 8, 16, 24, 32),
    (6, 1, 4, 8, 8)
])
def test_supernet_layer_sizes(batch_size, in_channels, out_channels, width, height):

    layer = SupernetLayer({
        'conv3x3': torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)),
        'conv5x5': torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5)),
        'conv7x7': torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(7, 7))
    })

    input_size = torch.Size((batch_size, in_channels, width, height))
    output_sizes = layer.get_output_sizes(input_size)
    expected_sizes = {
        'conv3x3': torch.Size((batch_size, out_channels, (width - 2) // 2, (height - 2) // 2)),
        'conv5x5': torch.Size((batch_size, out_channels, (width - 4) // 2, (height - 4) // 2)),
        'conv7x7': torch.Size((batch_size, out_channels, (width - 6) // 2, (height - 6) // 2))
    }
    
    assert output_sizes == expected_sizes

def test_supernet_model_forward():

    model = SupernetClassifier(input_size=(32, 32),
                               num_input_channels=1,
                               num_output_layer_channels=64,
                               num_classes=2, 
                               supernet_layers=[
        SupernetLayer({
            'conv3x3': torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3)),
            'conv5x5': torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5))
        }),
        SupernetLayer({
            'conv3x3': torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            'conv5x5': torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        })
    ])

    test_data = torch.rand(4, 1, 32, 32)
    results = {}

    for configuration in model.get_available_configurations():
        model.reconfigure(configuration)
        results[tuple(configuration)] = model(test_data)

    for c1, c2 in itertools.combinations(results.keys(), 2):
        if c1 != c2:
            assert not torch.allclose(results[c1], results[c2])

def test_supernet_model_distill():

    model = SupernetClassifier(input_size=(32, 32),
                               num_input_channels=1,
                               num_output_layer_channels=64,
                               num_classes=2, 
                               supernet_layers=[
        SupernetLayer({
            'conv3x3': torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3)),
            'conv5x5': torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5))
        }),
        SupernetLayer({
            'conv3x3': torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            'conv5x5': torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        })
    ])

    for configuration in model.get_available_configurations():
        model.reconfigure(configuration)
        distilled_model = model.distill()

        test_data = torch.rand(4, 1, 32, 32)
        full_model_output = model(test_data)
        distilled_model_output = distilled_model(test_data)
        assert torch.allclose(full_model_output, distilled_model_output)

def test_supernet_from_spec():

    test_config = {
        'input_size': [28, 28],
        'num_input_channels': 3,
        'num_output_layer_channels': 8,
        'num_classes': 2,
        'supernet_layers': [
            {
                'out_channels': 4,
                'blocks': {
                    'conv3x3': {'kind': 'conv2d', 'kernel_size': 3},
                    'conv5x5': {'kind': 'conv2d', 'kernel_size': 5},
                }
            },
            {
                'out_channels': 8,
                'blocks': {
                    'conv3x3': {'kind': 'conv2d', 'kernel_size': 3},
                    'conv5x5': {'kind': 'conv2d', 'kernel_size': 5},
                }
            }
        ]
    }

    test_spec = SupernetClassifierSpec.from_dict(test_config)
    expected_configurations = {
        ('conv3x3', 'conv3x3'),
        ('conv3x3', 'conv5x5'),
        ('conv5x5', 'conv3x3'),
        ('conv5x5', 'conv5x5')
    }

    model = SupernetClassifier.from_spec(test_spec)
    available_configurations = set(model.get_available_configurations())

    assert isinstance(model, SupernetClassifier)
    assert available_configurations == expected_configurations


def test