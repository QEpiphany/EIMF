import torch
import torch.nn as nn
import torch.nn.functional as F
from math import isclose
import math

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


BACKOFF_PROB = 1e-10

class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling
    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.
    Attributes:
        - probs: the probability density of desired multinomial distribution
    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        assert abs(probs.sum().item() - 1) < 1e-5, 'The noise distribution must sum to 1'

        cpu_probs = probs.cpu()
        K = len(probs)

        self_prob = [0] * K
        self_alias = [0] * K

        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial
        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)

class NCELoss(nn.Module):
    def __init__(self,
                 noise,
                 noise_ratio=100,
                 norm_term='auto',
                 reduction='elementwise_mean',
                 per_word=False,
                 loss_type='nce',
                 beta = 0
                 ):
        super(NCELoss, self).__init__()

        self.update_noise(noise)
        self.noise_ratio = noise_ratio
        self.beta = beta
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term
        self.reduction = reduction
        self.per_word = per_word
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type

    def update_noise(self, noise):
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        renormed_probs = probs / probs.sum()

        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)

    def forward(self, target, input, embs, interests=None, loss_fn = None, *args, **kwargs):


        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':


            noise_samples = trans_to_cuda(torch.arange(embs.size(0))).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1) if self.noise_ratio == 1 else self.get_noise(batch, max_len)

            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            logit_target_in_model, logit_noise_in_model = self._get_logit(target, noise_samples, input, embs, *args, **kwargs)



            if self.loss_type == 'nce':
                if self.training:
                    loss = self.nce_loss(
                        logit_target_in_model, logit_noise_in_model,
                        logit_noise_in_noise, logit_target_in_noise,
                    )
                else:

                    loss = - logit_target_in_model
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )

            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError(
                    'loss type {} not implemented at {}'.format(
                        self.loss_type, current_stage
                    )
                )

        else:
            loss = self.ce_loss(target, *args, **kwargs)

        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        noise_size = (batch_size, max_len, self.noise_ratio)
        if self.per_word:
            noise_samples = self.alias.draw(*noise_size)
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*noise_size)

        noise_samples = noise_samples.contiguous()
        return noise_samples

    def _get_logit(self, target_idx, noise_idx,input, embs, *args, **kwargs):
        """Get the logits of NCE estimated probability for target and noise
        Both NCE and sampled softmax Loss are unchanged when the probabilities are scaled
        evenly, here we subtract the maximum value as in softmax, for numeric stability.
        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        target_logit, noise_logit = self.get_score(target_idx, noise_idx, input, embs, *args, **kwargs)

        target_logit = target_logit.sub(self.norm_term)
        noise_logit = noise_logit.sub(self.norm_term)
        return target_logit, noise_logit

    def get_score(self, target_idx, noise_idx, input, embs, *args, **kwargs):
        """Get the target and noise score
        Usually logits are used as score.
        This method should be override by inherit classes
        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        original_size = target_idx.size()


        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)

        target_batch = embs[target_idx]

        target_score = torch.sum(input * target_batch, dim=1)

        noise_batch = embs[noise_idx]
        noise_score = torch.matmul(
            input, noise_batch.t()
        )
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def ce_loss(self, target_idx, *args, **kwargs):

        raise NotImplementedError()

    def nce_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):


        logit_model = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        logit_noise = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        logit_true = logit_model - logit_noise - math.log(self.noise_ratio)

        label = torch.zeros_like(logit_model)
        label[:, :, 0] = 1

        loss = self.bce_with_logits(logit_true, label).sum(dim=2)
        return loss

    def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        ori_logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        logits = ori_logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()

        if self.beta == 0:
            loss = self.ce(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ).view_as(labels)

        if self.beta != 0:
            x = ori_logits.view(-1, ori_logits.size(-1))
            x = x - torch.max(x, dim = -1)[0].unsqueeze(-1)
            pos = torch.exp(x[:,0])
            neg = torch.exp(x[:,1:])
            imp = (self.beta * x[:,1:] -  torch.max(self.beta * x[:,1:],dim = -1)[0].unsqueeze(-1)).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            if torch.isinf(reweight_neg).any() or torch.isnan(reweight_neg).any():
                import pdb; pdb.set_trace()
            Ng = reweight_neg

            stable_logsoftmax = -(x[:,0] - torch.log(pos + Ng))
            loss = torch.unsqueeze(stable_logsoftmax, 1)

        return loss
    

def build_noise(number, args=None):
    if args.sample_prob == 0:
        return build_uniform_noise(number)
    if args.sample_prob == 1:
        return build_log_noise(number)

def build_log_noise(number):
    total = number
    freq = torch.Tensor([1.0] * number).cuda()
    noise = freq / total
    for i in range(number):
        noise[i] = (np.log(i + 2) - np.log(i + 1)) / np.log(number + 1)

    assert abs(noise.sum() - 1) < 0.001
    return noise

def build_uniform_noise(number):
    total = number
    freq = torch.Tensor([1.0] * number).cuda()
    noise = freq / total 
    assert abs(noise.sum() - 1) < 0.001
    return noise

import time
from torch.nn.init import xavier_uniform_, xavier_normal_

class BasicModel(nn.Module):

    def __init__(self, item_num, hidden_size, batch_size, seq_len=50, beta=0):
        super(BasicModel, self).__init__()
        self.name = 'base'
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.item_num = item_num
        self.seq_len = seq_len
        self.beta = beta
        self.embeddings = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.interest_num = 0


    def set_sampler(self, args):
        self.is_sampler = True
        if args.sampled_n == 0:
            self.is_sampler = False
            return

        self.sampled_n = args.sampled_n

        noise = build_noise(self.item_num, args)

        self.sample_loss = NCELoss(noise=noise,
                                       noise_ratio=self.sampled_n,
                                       norm_term=0,
                                       reduction='elementwise_mean',
                                       per_word=False,
                                       loss_type=args.sampled_loss,
                                       beta=self.beta,
                                       )

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            torch.nn.init.kaiming_normal_(weight)


    def read_out(self, user_eb, label_eb):
        atten = torch.matmul(user_eb, torch.reshape(label_eb, (-1, self.hidden_size, 1)))
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, self.interest_num)), 1), dim=-1)

        if self.hard_readout:
            readout = torch.reshape(user_eb, (-1, self.hidden_size))[
                (torch.argmax(atten, dim=-1) + trans_to_cuda(
                    torch.arange(label_eb.shape[0])) * self.interest_num).long()]
        else:
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, self.interest_num)), user_eb)
            readout = torch.reshape(readout, (label_eb.shape[0], self.hidden_size))
        selection = torch.argmax(atten, dim=-1)
        return readout, selection

    def read_out_text(self, user_eb, label_eb):
        interest_num = user_eb.shape[1]
        atten = torch.matmul(user_eb, torch.reshape(label_eb, (-1, user_eb.shape[2], 1)))
        atten = F.softmax(torch.pow(torch.reshape(atten, (-1, interest_num)), 1), dim=-1)
        if self.hard_readout:
            readout = torch.reshape(user_eb, (-1, user_eb.shape[2]))[
                        (torch.argmax(atten, dim=-1) + trans_to_cuda(torch.arange(label_eb.shape[0])) * interest_num).long()]
        else:
            readout = torch.matmul(torch.reshape(atten, (label_eb.shape[0], 1, interest_num)), user_eb)
            readout = torch.reshape(readout, (label_eb.shape[0], user_eb.shape[2]))
        selection = torch.argmax(atten, dim=-1)
        return readout, selection

    def calculate_score(self, user_eb):
        all_items = self.embeddings.weight
        scores = torch.matmul(user_eb, all_items.transpose(1, 0))
        return scores


    def output_items(self):
        return self.embeddings.weight

    def calculate_full_loss(self, loss_fn, scores, target, interests):
        return loss_fn(scores, target)


    def calculate_sampled_loss(self, readout, pos_items, selection, interests):
        return self.sample_loss(pos_items.unsqueeze(-1), readout, self.embeddings.weight)

    def calculate_sampled_loss2(self, readout, pos_items, selection, interests, cluster_loss):
        loss = self.sample_loss(pos_items.unsqueeze(-1), readout, self.embeddings.weight)
        loss += cluster_loss * 0.1
        return loss


import numpy as np
import random
import math
class LogUniformSampler(object):
    def __init__(self, ntokens):

        self.N = ntokens
        self.prob = [0] * self.N

        self.generate_distribution()
        self.prob_tensor = torch.tensor(self.prob)
        self.cans = torch.arange(0, self.N)

    def generate_distribution(self):
        for i in range(self.N):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / np.log(self.N + 1)

    def probability(self, idx):
        return self.prob[idx]

    def expected_count(self, num_tries, samples):
        freq = list()
        for sample_idx in samples:
            freq.append(-(np.exp(num_tries * np.log(1-self.prob[sample_idx]))-1))
        return freq

    def accidental_match(self, labels, samples):
        sample_dict = dict()

        for idx in range(len(samples)):
            sample_dict[samples[idx]] = idx

        result = list()
        for idx in range(len(labels)):
            if labels[idx] in sample_dict:
                result.append((idx, sample_dict[labels[idx]]))

        return result

    def sample(self, size, labels):
        log_N = np.log(self.N)

        x = np.random.uniform(low=0.0, high=1.0, size=size)
        value = np.floor(np.exp(x * log_N)).astype(int) - 1
        samples = value.tolist()

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)
        if random.random() < 0.0002:
            print('By softmax', [round(i, 3) for i in true_freq], [round(i, 3) for i in sample_freq])

        return samples, true_freq, sample_freq

    def sample_uniform_prob(self, size, labels):
        idx = self.prob_tensor.multinomial(num_samples=size, replacement=False)
        b = self.cans[idx]

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, b)
        if random.random() < 0.0002:
            print('By uniform prob', [round(i, 3) for i in true_freq], [round(i, 3) for i in sample_freq])

        return b, true_freq, sample_freq

    def sample_uniform(self, size, labels):
        indice = random.sample(range(self.N), size)
        indice = torch.tensor(indice)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, indice)

        return indice, true_freq, sample_freq

    def sample_unique(self, size, labels):
        log_N = np.log(self.N)
        samples = list()

        while (len(samples) < size):
            x = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            value = np.floor(np.exp(x * log_N)).astype(int) - 1
            if value in samples:
                continue
            else:
                samples.append(value)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq

class CapsuleNetwork(nn.Module):
    def __init__(self, hidden_size, seq_len, bilinear_type=2, interest_num=4, routing_times=3, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                nn.ReLU()
            )
        if self.bilinear_type == 0:
            self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.hidden_size, self.hidden_size * self.interest_num, bias=False)
        else:
            self.w = nn.Parameter(torch.Tensor(1, self.seq_len, self.interest_num * self.hidden_size, self.hidden_size))
        

    def forward(self, item_eb, mask):
        if self.bilinear_type == 0:
            item_eb_hat = self.linear(item_eb)
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:
            u = torch.unsqueeze(item_eb, dim=2)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u, dim=3)
        
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.hidden_size))

        if self.stop_grad:
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        if self.bilinear_type > 0:
            capsule_weight = trans_to_cuda(torch.zeros(item_eb_hat.shape[0], self.interest_num, self.seq_len, requires_grad=False))
        else:
            capsule_weight = trans_to_cuda(torch.randn(item_eb_hat.shape[0], self.interest_num, self.seq_len, requires_grad=False))

        for i in range(self.routing_times):
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_eb_hat_iter,
                                        torch.transpose(interest_capsule, 2, 3).contiguous()
                                        )
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.hidden_size))

        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)
        
        return interest_capsule
