from torch import nn


class ForwardHook:
    def __init__(self,
                 module: nn.Module,
                 multi=False):
        self.multi = multi
        self.hooked_input = None
        self.hooked_output = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self,
                module: nn.Module,
                module_input,
                module_output):
        self.hooked_input = [_in for _in in module_input]
        if self.multi:
            self.hooked_output = [_out for _out in module_output]
        else:
            self.hooked_output = module_output

    def close(self):
        self.hook.remove()

    def get_hooked_input(self):
        return self.hooked_input

    def get_hooked_output(self):
        return self.hooked_output
