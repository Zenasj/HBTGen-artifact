import torch
import torch.distributed as dist

class PrefixStoreExample:
    def __init__(self):
        # Initialize PrefixStore without providing a prefix
        self.prefix_store = dist.PrefixStore('my_prefix', None)

    def add_and_get_key(self):
        # Add key-value pairs
        self.prefix_store.set('some_key', 'some_value')

        # Try to get a non-existent key that has not been prefixed correctly
        key = 'my_prefix_never_exists'
        try:
            value = self.prefix_store.get(key)
            print(f"The value of '{key}' is {value}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")


if __name__ == '__main__':
    store_example = PrefixStoreExample()
    store_example.add_and_get_key()