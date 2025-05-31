class FlowModel:
    """
    Abstract base class for flow models.
    """
    def do_flow(self, z):
        raise NotImplementedError("decode() must be implemented by subclasses.")
        