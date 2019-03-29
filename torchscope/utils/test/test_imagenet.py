import pytest

from torchscope.utils import ImageNetIndex


@pytest.fixture
def imagenet():
    return ImageNetIndex()


def test_list_all_classes(imagenet):
    assert len(imagenet.keys()) == 1000


def test_return_true_when_target_class_exists(imagenet):
    assert ('dalmatian' in imagenet) == True


def test_return_false_when_target_class_does_not_exist(imagenet):
    assert ('invalid class' in imagenet) == False


def test_find_class_index(imagenet):
    class_index = imagenet['dalmatian']

    assert class_index == 251


def test_handle_multi_word_target_class(imagenet):
    class_index = imagenet['dalmatian dog']

    assert class_index == 251


def test_handle_partial_match(imagenet):
    class_index = imagenet['foxhound']

    assert class_index == 167


def test_raise_on_invalid_argument_type(imagenet):
    with pytest.raises(TypeError) as error:
        class_index = imagenet[1]

    assert 'Target class needs to be a string' in str(error.value)


def test_raise_on_invalid_class_name(imagenet):
    with pytest.raises(ValueError) as error:
        class_index = imagenet['invalid class name']

    assert 'Cannot find the specified class' in str(error.value)


def test_raise_on_multiple_matches(imagenet):
    with pytest.raises(ValueError) as error:
        class_index = imagenet['dog']

    assert 'Multiple matches found' in str(error.value)
