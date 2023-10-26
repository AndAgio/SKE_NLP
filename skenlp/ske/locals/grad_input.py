from .grad_sensitivity import GradSensitivity


class GradInput(GradSensitivity):
    def __init__(self,
                 tokenizer,
                 model,
                 multi_label=False,
                 threshold=0.,
                 device='cpu',
                 out_dir='outs',
                 logger=None):
        super().__init__(tokenizer, model, multi_label, threshold, device, out_dir, logger)
        self.explainer_name = 'gi'
        self.multiply = True
