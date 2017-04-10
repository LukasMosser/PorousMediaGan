import h5py

def save_hdf5(tensor, filename):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    tensor = tensor.cpu()
    ndarr = tensor.mul(0.5).add(0.5).mul(255).byte().numpy()#.transpose(0,2).transpose(0,1).numpy()
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=ndarr, dtype="i8", compression="gzip")