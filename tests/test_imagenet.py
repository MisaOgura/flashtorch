import pytest
from collections.abc import Iterable

from flashtorch.utils import ImageNetIndex


#################
# Test fixtures #
#################


@pytest.fixture
def imagenet():
    return ImageNetIndex()


def test_is_iterable(imagenet):
    assert isinstance(imagenet, Iterable)
    assert isinstance(iter(imagenet), Iterable)


def test_return_length(imagenet):
    assert len(imagenet) == 1000


def test_list_keys(imagenet):
    assert len(imagenet.keys()) == 1000


def test_list_items(imagenet):
    assert len(imagenet.items()) == 1000


def test_return_true_when_target_class_exists(imagenet):
    assert ('dalmatian' in imagenet) == True


def test_return_false_when_target_class_does_not_exist(imagenet):
    assert ('invalid class' in imagenet) == False


def test_find_class_index(imagenet):
    class_index = imagenet['jay']

    assert class_index == 17


def test_find_whole_match_first(imagenet):
    class_index = imagenet['king penguin']

    assert class_index == 145


def test_handle_multi_word_target_class(imagenet):
    class_index = imagenet['dalmatian dog']

    assert class_index == 251

def test_handle_partial_match(imagenet):
    class_index = imagenet['foxhound']

    assert class_index == 167


def test_return_none_for_invalid_class_name(imagenet):
    class_index = imagenet['invalid class name']

    assert class_index == None


def test_raise_on_invalid_argument_type(imagenet):
    with pytest.raises(TypeError) as error:
        class_index = imagenet[1]

    assert 'Target class needs to be a string' in str(error.value)


def test_raise_on_multiple_matches(imagenet):
    with pytest.raises(ValueError) as error:
        class_index = imagenet['dog']

    assert 'Multiple potential matches found' in str(error.value)


if __name__ == '__main__':
    pytest.main([__file__])
