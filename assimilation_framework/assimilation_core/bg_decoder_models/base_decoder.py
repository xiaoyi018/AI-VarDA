class BackgroundDecoder:
    """
    Abstract base class for background decoders.
    Used to transform control variables (z) into physical state variables (x).
    """
    def decode(self, z):
        raise NotImplementedError("decode() must be implemented by subclasses.")
        