import torch

from .base import BaseLocalExplainer
from .hooks import ForwardHook


class LayerwiseRelevancePropagation(BaseLocalExplainer):
    def __init__(self,
                 tokenizer,
                 model,
                 multi_label=False,
                 threshold=0.,
                 device='cpu',
                 out_dir='outs',
                 logger=None):
        super().__init__(tokenizer, model, multi_label, threshold, device, out_dir, logger)
        self.hooks = {}
        for name, children_module in self.model.named_modules():
            if name[-15:] == '.attention.self':
                self.hooks[name] = ForwardHook(children_module,
                                               multi=True)
            else:
                self.hooks[name] = ForwardHook(children_module)
        self.device = device
        self.explainer_name = 'lrp'

    def explain(self, sentence, label=None, sentence_id=0, dataset_config=None,
                visualize=True, verbose=False, get_pred_labels=False):
        tokens, outputs, pred_labels = self.feed_input(sentence)
        r_out_mask = torch.zeros((torch.tensor(tokens['input_ids']).shape[0],
                                  self.model.n_labels)).to(self.device)
        r_out_mask[:, pred_labels] = 1.0
        relevance_score = outputs * r_out_mask
        if verbose:
            self.logger.print_it('Computing lrp. This may take a while...')
        lrp_score = self.backward_lrp(relevance_score)
        lrp_score = lrp_score.cpu().detach().data
        lrp_score = torch.abs(lrp_score).sum(dim=-1)
        lrp_score = torch.squeeze(lrp_score)

        words, lrp_score = self.refine_tokens_and_scores(tokens, lrp_score)
        assert len(words) == lrp_score.shape[0]

        if verbose:
            self.logger.print_it('lrp_score: {}'.format(lrp_score))
            self.logger.print_it('lrp_score.shape: {}'.format(lrp_score.shape))
        if visualize:
            self.visualize(scores=lrp_score,
                           outputs=outputs,
                           pred_labels=pred_labels,
                           label=label,
                           words=words,
                           sentence=sentence,
                           sentence_id=sentence_id,
                           dataset_config=dataset_config)
        if get_pred_labels:
            return words, lrp_score, pred_labels
        else:
            return words, lrp_score

    def backward_lrp(self, relevance_score):
        classifier_in = self.hooks['classifier'].get_hooked_input()[0]
        classifier_out = self.hooks['classifier'].get_hooked_output()
        relevance_score = LayerwiseRelevancePropagation.backprop_lrp_fc(self.model.classifier.weight,
                                                                        self.model.classifier.bias,
                                                                        classifier_in,
                                                                        relevance_score)
        relevance_score = self.backward_lrp_bert(relevance_score)
        return relevance_score

    def backward_lrp_bert(self, relevance_score):
        relevance_score = self.backward_lrp_pooler(relevance_score)
        relevance_score = self.backward_lrp_encoder(relevance_score)
        return relevance_score

    def backward_lrp_pooler(self, relevance_score):
        dense_in = self.hooks['bert.pooler.dense'].get_hooked_input()[0]  # func_inputs['model.bert.pooler.dense'][0]
        relevance_score = LayerwiseRelevancePropagation.backprop_lrp_fc(self.model.bert.pooler.dense.weight,
                                                                        self.model.bert.pooler.dense.bias,
                                                                        dense_in,
                                                                        relevance_score)
        # we need to scatter this to all hidden states, but only first one matters!
        pooler_in = self.hooks['bert.pooler'].get_hooked_input()[0]  # func_inputs['model.bert.pooler'][0]
        relevance_score_all = torch.zeros_like(pooler_in)
        relevance_score_all[:, 0] = relevance_score
        return relevance_score_all

    def backward_lrp_encoder(self, relevance_score):
        # backout layer by layer from last to the first
        layer_module_index = self.model.config.num_hidden_layers - 1
        for layer_module in reversed(self.model.bert.encoder.layer):
            relevance_score = self.backward_lrp_bert_layer(layer_module, relevance_score, layer_module_index)
            layer_module_index -= 1
        return relevance_score

    def backward_lrp_bert_layer(self, layer_module, relevance_score, layer_module_index):
        relevance_score, relevance_score_residual = self.backward_lrp_bert_layer_output(layer_module.output,
                                                                                        relevance_score,
                                                                                        layer_module_index)
        relevance_score = self.backward_lrp_bert_layer_intermediate(layer_module.intermediate,
                                                                    relevance_score,
                                                                    layer_module_index)
        # merge
        relevance_score += relevance_score_residual
        relevance_score = self.backward_lrp_bert_layer_attention(layer_module.attention,
                                                                 relevance_score,
                                                                 layer_module_index)
        return relevance_score

    def backward_lrp_bert_layer_output(self, layer_output, relevance_score, layer_module_index):
        # residual conection handler
        layer_name = 'bert.encoder.layer.{}.output'.format(layer_module_index)
        output_in_input = self.hooks[layer_name].get_hooked_input()[1]  # func_inputs[layer_name][1]
        output_out = self.hooks[layer_name].get_hooked_output()  # func_activations[layer_name]
        relevance_score_residual = torch.autograd.grad(output_out, output_in_input,
                                                       grad_outputs=relevance_score,
                                                       retain_graph=True)[0]
        # main connection
        layer_name_dense = 'bert.encoder.layer.{}.output.dense'.format(layer_module_index)
        dense_out = self.hooks[layer_name_dense].get_hooked_output()  # func_activations[layer_name_dense]
        relevance_score = torch.autograd.grad(output_out, dense_out,
                                              grad_outputs=relevance_score,
                                              retain_graph=True)[0]
        dense_in = self.hooks[layer_name_dense].get_hooked_input()[0]  # func_inputs[layer_name_dense][0]
        relevance_score = LayerwiseRelevancePropagation.backprop_lrp_fc(layer_output.dense.weight,
                                                                        layer_output.dense.bias,
                                                                        dense_in,
                                                                        relevance_score)
        return relevance_score, relevance_score_residual

    def backward_lrp_bert_layer_intermediate(self, layer_intermediate, relevance_score, layer_module_index):
        layer_name = 'bert.encoder.layer.{}.intermediate.dense'.format(layer_module_index)
        dense_in = self.hooks[layer_name].get_hooked_input()[0]  # func_inputs[layer_name][0]
        relevance_score = LayerwiseRelevancePropagation.backprop_lrp_fc(layer_intermediate.dense.weight,
                                                                        layer_intermediate.dense.bias,
                                                                        dense_in,
                                                                        relevance_score)
        return relevance_score

    def backward_lrp_bert_layer_attention(self, layer_attention, relevance_score, layer_module_index):
        relevance_score, relevance_score_residual = self.backward_lrp_bert_layer_attention_output(
            layer_attention.output,
            relevance_score,
            layer_module_index)
        relevance_score = self.backward_lrp_bert_layer_attention_self(layer_attention.self,
                                                                      relevance_score,
                                                                      layer_module_index)
        # merge
        relevance_score = relevance_score + relevance_score_residual
        return relevance_score

    def backward_lrp_bert_layer_attention_output(self, layer_output, relevance_score, layer_module_index):
        # residual conection handler
        layer_name = 'bert.encoder.layer.{}.attention.output'.format(layer_module_index)
        output_in_input = self.hooks[layer_name].get_hooked_input()[1]  # func_inputs[layer_name][1]
        output_out = self.hooks[layer_name].get_hooked_output()  # func_activations[layer_name]
        relevance_score_residual = torch.autograd.grad(output_out, output_in_input,
                                                       grad_outputs=relevance_score,
                                                       retain_graph=True)[0]
        # main connection
        layer_name_dense = 'bert.encoder.layer.{}.attention.output.dense'.format(layer_module_index)
        dense_out = self.hooks[layer_name_dense].get_hooked_output()  # func_activations[layer_name_dense]
        relevance_score = torch.autograd.grad(output_out, dense_out,
                                              grad_outputs=relevance_score,
                                              retain_graph=True)[0]
        dense_in = self.hooks[layer_name_dense].get_hooked_input()[0]  # func_inputs[layer_name_dense][0]
        relevance_score = LayerwiseRelevancePropagation.backprop_lrp_fc(layer_output.dense.weight,
                                                                        layer_output.dense.bias,
                                                                        dense_in,
                                                                        relevance_score)
        return relevance_score, relevance_score_residual

    def backward_lrp_bert_layer_attention_self(self, layer_self, relevance_score, layer_module_index):
        layer_name_value = 'bert.encoder.layer.{}.attention.self.value'.format(layer_module_index)
        layer_name_query = 'bert.encoder.layer.{}.attention.self.query'.format(layer_module_index)
        layer_name_key = 'bert.encoder.layer.{}.attention.self.key'.format(layer_module_index)
        value_in = self.hooks[layer_name_value].get_hooked_input()[0]  # func_inputs[layer_name_value][0]
        value_out = self.hooks[layer_name_value].get_hooked_output()  # func_activations[layer_name_value]
        query_in = self.hooks[layer_name_query].get_hooked_input()[0]  # func_inputs[layer_name_query][0]
        query_out = self.hooks[layer_name_query].get_hooked_output()  # func_activations[layer_name_query]
        key_in = self.hooks[layer_name_key].get_hooked_input()[0]  # func_inputs[layer_name_key][0]
        key_out = self.hooks[layer_name_key].get_hooked_output()  # func_activations[layer_name_key]
        layer_name_self = 'bert.encoder.layer.{}.attention.self'.format(layer_module_index)
        context_layer = self.hooks[layer_name_self].get_hooked_output()[0]  # func_activations[layer_name_self][0]
        attention_mask = self.hooks[layer_name_self].get_hooked_input()[1]  # func_inputs[layer_name_self][1]
        # Instead of jacobian, we may estimate this using a linear layer.
        # This turns out to be a good estimate in general.
        relevance_query = torch.autograd.grad(context_layer,
                                              query_out,
                                              grad_outputs=relevance_score,
                                              retain_graph=True)[0]
        relevance_key = torch.autograd.grad(context_layer,
                                            key_out,
                                            grad_outputs=relevance_score,
                                            retain_graph=True)[0]
        relevance_value = torch.autograd.grad(context_layer,
                                              value_out,
                                              grad_outputs=relevance_score,
                                              retain_graph=True)[0]
        relevance_query = LayerwiseRelevancePropagation.backprop_lrp_fc(layer_self.query.weight,
                                                                        layer_self.query.bias,
                                                                        query_in,
                                                                        relevance_query)
        relevance_key = LayerwiseRelevancePropagation.backprop_lrp_fc(layer_self.key.weight,
                                                                      layer_self.key.bias,
                                                                      key_in,
                                                                      relevance_key)
        relevance_value = LayerwiseRelevancePropagation.backprop_lrp_fc(layer_self.value.weight,
                                                                        layer_self.value.bias,
                                                                        value_in,
                                                                        relevance_value)
        relevance_score = relevance_query + relevance_key + relevance_value
        return relevance_score

    @staticmethod
    def rescale_lrp(post_a: torch.tensor,
                    inp_relevance: torch.tensor) -> torch.tensor:
        inp_relevance = torch.abs(inp_relevance)
        if len(post_a.shape) == 2:
            ref_scale = torch.sum(post_a, dim=-1, keepdim=True) + 1e-7
            inp_scale = torch.sum(inp_relevance, dim=-1, keepdim=True) + 1e-7
        elif len(post_a.shape) == 3:
            ref_scale = post_a.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True) + 1e-7
            inp_scale = inp_relevance.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True) + 1e-7
        else:
            raise ValueError('Something very wrong with LRP rescaling shapes...')
        scaler = ref_scale / inp_scale
        inp_relevance = inp_relevance * scaler
        return inp_relevance

    @staticmethod
    def backprop_lrp_fc(weight, bias, activations, r,
                        eps=1e-7, alpha=0.5):
        beta = 1.0 - alpha

        weight_p = torch.clamp(weight, min=0.0)
        bias_p = torch.clamp(bias, min=0.0)
        z_p = torch.matmul(activations, weight_p.T) + bias_p + eps
        s_p = r / z_p
        c_p = torch.matmul(s_p, weight_p)

        weight_n = torch.clamp(weight, max=0.0)
        bias_n = torch.clamp(bias, max=0.0)
        z_n = torch.matmul(activations, weight_n.T) + bias_n - eps
        s_n = r / z_n
        c_n = torch.matmul(s_n, weight_n)

        r_c = activations * (alpha * c_p + beta * c_n)

        r_c = LayerwiseRelevancePropagation.rescale_lrp(r, r_c)

        return r_c
