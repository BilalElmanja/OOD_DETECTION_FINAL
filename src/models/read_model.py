import pickle

# Path to the .pkl file
file_path = '../../models/ImageNet-1K/archive/data.pkl'


class CustomUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        typename = pid[0]
        if typename == 'storage':
            # Handle storage objects from PyTorch
            storage_type = pid[1]
            root_key = pid[2]
            location = pid[3]
            numel = pid[4]
            # Reconstruct the storage
            storage = storage_type._new_with_file(root_key, numel)
            storage._untyped_storage = storage
            return storage
        else:
            raise pickle.UnpicklingError("unsupported persistent identifier: %r" % (pid,))

# Path to the .pkl file

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Create an instance of the custom unpickler
    unpickler = CustomUnpickler(file)
    # Load the data using the custom unpickler
    data = unpickler.load()

# Now 'data' contains the deserialized object
print(data)
