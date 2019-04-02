class WavedataProvider(object):
    @staticmethod
    def save(slices, directory):
        """Save slices to disk in a proper file format
        Args:
            slices: list of Slice namedpuples
            directory: directory to save slices
        """
        raise NotImplementedError()

    @staticmethod
    def load(directory, include_augmented=False):
        """Loads examples from disk
        Args:
            directory: target directory
            include_augmented: if True, return augmented images as secod element of returned list
        Returns:
            ([regular_examples], <[augmented_examples]>), elemets are Example naedpuples
        """
        raise NotImplementedError()