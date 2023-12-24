import keras
class MyGAOptimizer(keras.optimizers.Optimizer):
    def __init__(self, name="MyGAOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        # self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))  # handle lr=learning_rate
        # self._set_hyper("decay", self._initial_decay)  #
        # self._set_hyper("momentum", momentum)
    #def _create_s

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return

    def update_step(self, gradient, variable):

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }