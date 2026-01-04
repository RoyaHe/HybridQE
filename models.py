from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import os
from tqdm import tqdm

query_only2query_structure = {
     ('e', ('r', 'o')) : ('e', ('r',)),
     ('e', ('r', 'r', 'o')) : ('e', ('r', 'r')),
     ('e', ('r', 'r', 'r', 'o')) : ('e', ('r', 'r', 'r')),
     (('e', ('r',)), ('e', ('r',)), 'o') : (('e', ('r',)), ('e', ('r',))),
     (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), 'o') : (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
     ('e', ('r', 'n', 'o')) : ('e', ('r', 'n')),
     (('e', ('r',)), ('e', ('r', 'n')), 'o') : (('e', ('r',)), ('e', ('r', 'n'))),
     (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')), 'o') : (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))),
     (('e', ('r',)), ('e', ('r',)), ('u',), 'o') : (('e', ('r',)), ('e', ('r',)), ('u',)),
     (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',), 'o') : (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('u',))
     }

def Identity(x):
    return x

class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings)) # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0) # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim) # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim) # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x

class GammaIntersection(nn.Module):

    def __init__(self, dim):
        super(GammaIntersection, self).__init__()
        self.dim = dim
        self.layer_alpha1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_beta1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_alpha2 = nn.Linear(self.dim, self.dim)
        self.layer_beta2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_alpha1.weight)
        nn.init.xavier_uniform_(self.layer_beta1.weight)
        nn.init.xavier_uniform_(self.layer_alpha2.weight)
        nn.init.xavier_uniform_(self.layer_beta2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_alpha = F.relu(self.layer_alpha1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        attention1 = F.softmax(self.layer_alpha2(layer1_alpha), dim=0)  # (num_conj, batch_size, dim)

        layer1_beta = F.relu(self.layer_beta1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        attention2 = F.softmax(self.layer_beta2(layer1_beta), dim=0)  # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention1 * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention2 * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding


class GammaUnion(nn.Module):
    def __init__(self, dim, projection_regularizer, drop):
        super(GammaUnion, self).__init__()
        self.dim = dim
        self.layer_alpha1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_beta1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_alpha2 = nn.Linear(self.dim, self.dim // 2)
        self.layer_beta2 = nn.Linear(self.dim, self.dim // 2)
        self.layer_alpha3 = nn.Linear(self.dim // 2, self.dim)
        self.layer_beta3 = nn.Linear(self.dim // 2, self.dim)

        self.projection_regularizer = projection_regularizer
        self.drop = nn.Dropout(p=drop)
        nn.init.xavier_uniform_(self.layer_alpha1.weight)
        nn.init.xavier_uniform_(self.layer_beta1.weight)
        nn.init.xavier_uniform_(self.layer_alpha2.weight)
        nn.init.xavier_uniform_(self.layer_beta2.weight)
        nn.init.xavier_uniform_(self.layer_alpha3.weight)
        nn.init.xavier_uniform_(self.layer_beta3.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_alpha = F.relu(self.layer_alpha1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        layer2_alpha = F.relu(self.layer_alpha2(layer1_alpha))
        attention1 = F.softmax(self.drop(self.layer_alpha3(layer2_alpha)), dim=0)  # (num_conj, batch_size, dim)

        layer1_beta = F.relu(self.layer_beta1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        layer2_beta = F.relu(self.layer_beta2(layer1_beta))
        attention2 = F.softmax(self.drop(self.layer_beta3(layer2_beta)), dim=0)  # (num_conj, batch_size, dim)

        k = alpha_embeddings * attention1
        o = 1 / (beta_embeddings * attention2)
        k_sum = torch.pow(torch.sum(k * o, dim=0), 2) / torch.sum(torch.pow(o, 2) * k, dim=0)
        o_sum = torch.sum(k * o, dim=0) / (k_sum * o.shape[0])
        # Welch–Satterthwaite equation
        
        alpha_embedding = k_sum
        beta_embedding = o_sum
        alpha_embedding[torch.abs(alpha_embedding) < 1e-4] = 1e-4
        beta_embedding[torch.abs(beta_embedding) < 1e-4] = 1e-4
        return alpha_embedding, beta_embedding


class GammaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(GammaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

        self.layerr1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layerr2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layerr3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layerr0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layerr{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layerr{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, alpha_embedding, beta_embedding, alpha_embedding_r, beta_embedding_r):

        xa = torch.cat([alpha_embedding, alpha_embedding_r], dim=-1)
        xb = torch.cat([beta_embedding, beta_embedding_r], dim=-1)

        for nl in range(1, self.num_layers + 1):
            xa = F.relu(getattr(self, "layer{}".format(nl))(xa))
        xa = self.layer0(xa)
        xa = self.projection_regularizer(xa)
        for nl in range(1, self.num_layers + 1):
            xb = F.relu(getattr(self, "layerr{}".format(nl))(xb))
        xb = self.layerr0(xb)
        xb = self.projection_regularizer(xb)

        alpha_embeddings = xa
        beta_embeddings = xb

        return alpha_embeddings, beta_embeddings

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class KGReasoning(nn.Module):
    def __init__(self, nprod, nentity, nrelation,
        hidden_dim, gamma, geo, use_cuda=False, center_reg=None, 
        beta_mode=None, gamma_mode=None, test_batch_size=1, params_frozen=False,
        query_name_dict=None, drop=0.0, graph_type=None, ent_bert_embeddings=None, rel_bert_embeddings=None, textual_query_embeddings=None,
        textual_symbolic_query_embeddings=None):
        super(KGReasoning, self).__init__()

        self.nprod = nprod
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.graph_type = graph_type
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict
        
        '''
        self.ent_bert_embeddings = nn.Embedding(ent_bert_embeddings.shape[0], ent_bert_embeddings.shape[1])
        self.rel_bert_embeddings = nn.Embedding(rel_bert_embeddings.shape[0], rel_bert_embeddings.shape[1])
        self.ent_bert_embeddings.weight = nn.Parameter(ent_bert_embeddings)
        self.rel_bert_embeddings.weight = nn.Parameter(rel_bert_embeddings)

        if params_frozen:
            # Freeze the embedding weights
            self.ent_bert_embeddings.weight.requires_grad = False
            self.rel_bert_embeddings.weight.requires_grad = True
        else:
            self.ent_bert_embeddings.weight.requires_grad = True
            self.rel_bert_embeddings.weight.requires_grad = True
        '''
        self.textual_query_embeddings = nn.Parameter(textual_query_embeddings, requires_grad=False) 
        self.textual_symbolic_query_embeddings = nn.Parameter(textual_symbolic_query_embeddings, requires_grad=False)

        if params_frozen:
            self.ent_bert_embeddings = nn.Parameter(ent_bert_embeddings, requires_grad=False)
            self.rel_bert_embeddings = nn.Parameter(rel_bert_embeddings, requires_grad=True)
        else:
            self.ent_bert_embeddings = nn.Parameter(ent_bert_embeddings, requires_grad=True)
            self.rel_bert_embeddings = nn.Parameter(rel_bert_embeddings, requires_grad=True)

        #self.ent_bert_embeddings = ent_bert_embeddings
        #self.rel_bert_embeddings = rel_bert_embeddings
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        D_in, H, D_out = 768, 768, hidden_dim
        self.ent_linear = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(),nn.Dropout(), nn.Linear(H, 2*D_out))
        #self.ent_linear = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(),nn.Dropout(), nn.Linear(H, H), nn.ReLU(),nn.Dropout(), nn.Linear(H, H), nn.ReLU(),nn.Dropout(), nn.Linear(H, 2*D_out))
        self.rel_linear = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(),nn.Dropout(), nn.Linear(H, D_out))
        for param in self.ent_linear.parameters():
            param.requires_grad = True
        for param in self.rel_linear.parameters():
            param.requires_grad = True
        self.linear_gate_text = nn.Linear(768, 1)
        self.linear_gate_sym = nn.Linear(768, 1)

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if self.geo == 'beta':
            self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive
           #self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2)) # alpha and beta
        elif self.geo == 'gamma':
            self.ent_linear = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(),nn.Dropout(), nn.Linear(H, 2*D_out))
            self.rel_linear = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(),nn.Dropout(), nn.Linear(H, 2*D_out))
            self.entity_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05, 1e9) # make sure the parameters of beta embeddings after relation projection are positive
        
        '''
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        '''

        if self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2, 
                                             self.relation_dim, 
                                             hidden_dim, 
                                             self.projection_regularizer, 
                                             num_layers)
        elif self.geo == 'gamma':
            hidden_dim, num_layers = gamma_mode
            self.nentity = nentity
            self.nrelation = nrelation
            self.hidden_dim = hidden_dim
            self.epsilon = 2.0
            self.geo = geo
            self.is_u = False
            self.use_cuda = use_cuda
            self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size,
                                                                               1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
            self.entity_regularizer = Regularizer(1, 0.15, 1e9)  # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.15, 1e9)
            self.modulus = nn.Parameter(torch.Tensor([1 * self.embedding_range.item()]), requires_grad=True)
            self.center_net = GammaIntersection(self.entity_dim)
            self.projection_net = GammaProjection(self.entity_dim,
                                                self.relation_dim,
                                                hidden_dim,
                                                self.projection_regularizer,
                                                num_layers)
            self.union_net = GammaUnion(self.entity_dim, self.projection_regularizer, drop)

            
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_symbolic_queries_dict, batch_text_queries_dict, batch_text_symbolic_query_idx, batch_idxs_dict):
        if self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_symbolic_queries_dict, batch_text_queries_dict, batch_text_symbolic_query_idx, batch_idxs_dict)
        elif self.geo == 'gamma':
            return self.forward_gamma(positive_sample, negative_sample, subsampling_weight, batch_symbolic_queries_dict, batch_text_queries_dict, batch_text_symbolic_query_idx, batch_idxs_dict)
   
    def transform_gamma_union_query(self, queries, query_structure):
        if self.query_name_dict[query_structure] == '2u':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == '3u':
            queries = queries[:, :-1] # remove union -1
        return queries

    def transform_gamma_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u':
            return (('e', ('r',)), ('e', ('r',)))
        if self.query_name_dict[query_structure] == '3u':
            return (('e', ('r',)), ('e', ('r',)), ('e', ('r',)))
        
    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if '2u' in self.query_name_dict[query_structure]:
            queries = queries[:, :-1] # remove union -1
            queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        elif '3u' in self.query_name_dict[query_structure]:
            queries = queries[:, :-1] # remove union -1
            queries = torch.reshape(queries, [queries.shape[0]*3, -1])
        
        return queries

    def transform_union_text_query(self, text_queries, query_structure):
        
        if '2u' in self.query_name_dict[query_structure]:
            text_queries = text_queries.repeat(2)
        elif '3u' in self.query_name_dict[query_structure]:
            text_queries = text_queries.repeat(3)
        
        return text_queries
    
    def transform_union_text_symbolic_query_idx(self, text_symbolic_query_idx, query_structure):

        if '2u' in self.query_name_dict[query_structure]:
            text_symbolic_query_idx = text_symbolic_query_idx.repeat(2)
        elif '3u' in self.query_name_dict[query_structure]:
            text_symbolic_query_idx = text_symbolic_query_idx.repeat(3)
        
        return text_symbolic_query_idx

    def transform_union_structure(self, query_structure):
        if '2u' in self.query_name_dict[query_structure]:
            return ('e', ('r',))
        if '3u' in self.query_name_dict[query_structure]:
            return ('e', ('r',))
        elif 'up' in self.query_name_dict[query_structure]:
            return ('e', ('r', 'r'))
    
    def embed_symbolic_query_beta(self, symbolic_queries, text_queries, text_symbolic_query_idx, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                # Compute attention scores (dot-product attention)
                scores_text = torch.bmm(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0), torch.index_select(self.textual_query_embeddings, dim=0, index=text_queries).transpose(0, 1).unsqueeze(0))
                attn_weights_text = torch.softmax(scores_text, dim=-1)
                attended_Qtext = torch.bmm(attn_weights_text, torch.index_select(self.textual_query_embeddings, dim=0, index=text_queries).unsqueeze(0))
                
                scores_sym = torch.bmm(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0), torch.index_select(self.textual_symbolic_query_embeddings, dim=0, index=text_symbolic_query_idx).transpose(0, 1).unsqueeze(0))
                attn_weights_sym = torch.softmax(scores_sym, dim=-1)
                attended_Qsym = torch.bmm(attn_weights_sym, torch.index_select(self.textual_symbolic_query_embeddings, dim=0, index=text_symbolic_query_idx).unsqueeze(0))
                
                gate_text = torch.sigmoid(self.linear_gate_text(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0) + attended_Qtext))
                gate_sym = torch.sigmoid(self.linear_gate_sym(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0) + attended_Qsym))
                ent_bert_embedding_ca = torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0) + gate_text * attended_Qtext + gate_sym * attended_Qsym

                embedding = self.entity_regularizer(self.ent_linear(ent_bert_embedding_ca.squeeze(0)))
                #embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]))) 
                # self.bert(input_ids=ent_title, attention_mask=ent_title_m)[0][:, 0, :]
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_symbolic_query_beta(symbolic_queries, text_queries, text_symbolic_query_idx, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (symbolic_queries[:, idx] == -2).all()
                    embedding = 1./embedding
                else:
                    if symbolic_queries[:,idx][0] == -3:
                        idx += 1
                    #rel = torch.index_select(self.id2rel, dim=0, index=symbolic_queries[:, idx])
                    #rel_m = torch.index_select(self.id2rel_m, dim=0, index=symbolic_queries[:, idx])
                    r_embedding = self.rel_linear(torch.index_select(self.rel_bert_embeddings, dim=0, index=symbolic_queries[:, idx]))
                    # self.bert(input_ids=rel, attention_mask=rel_m)[0][:, 0, :]
                    #r_embedding = torch.index_select(self.relation_embedding, dim=0, index=symbolic_queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            if query_structure[-1][-1] == 's':
                ## split
                alpha_embedding_list = []
                beta_embedding_list = []
                alpha_embedding, beta_embedding, idx = self.embed_symbolic_query_beta(symbolic_queries, text_queries, text_symbolic_query_idx, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
                for i in range(len(query_structure[-1])-1):
                    if len(query_structure[-1][i]) == 2:
                        ## negation
                        #assert (queries[:, idx] == -2).all()
                        embedding = 1./embedding
                    else:
                        #rel = torch.index_select(self.id2rel, dim=0, index=symbolic_queries[:, idx])
                        #rel_m = torch.index_select(self.id2rel_m, dim=0, index=symbolic_queries[:, idx])
                        r_embedding = self.rel_linear(torch.index_select(self.rel_bert_embeddings, dim=0, index=symbolic_queries[:, idx]))
                        # self.bert(input_ids=rel, attention_mask=rel_m)[0][:, 0, :]
                        #r_embedding = torch.index_select(self.relation_embedding, dim=0, index=symbolic_queries[:, idx])
                        embedding = self.projection_net(embedding, r_embedding)
                        idx += 1
                    alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
                    alpha_embedding_list.append(alpha_embedding)
                    beta_embedding_list.append(beta_embedding)
                alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))
            else:
                ## intersection
                alpha_embedding_list = []
                beta_embedding_list = []
                for i in range(len(query_structure)):
                    alpha_embedding, beta_embedding, idx = self.embed_symbolic_query_beta(symbolic_queries, text_queries, text_symbolic_query_idx, query_structure[i], idx)
                    alpha_embedding_list.append(alpha_embedding)
                    beta_embedding_list.append(beta_embedding)
                alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx
    
    def embed_text_query_beta(self, text_queries):
        embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.textual_query_embeddings, dim=0, index=text_queries)))
        #embedding = self.entity_regularizer(self.ent_linear(self.bert(input_ids=text_queries[:,0,:], attention_mask=text_queries[:,1,:])[0][:, 0, :]))
        alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        return alpha_embedding, beta_embedding

    def embed_hybrid_query_beta(self, symbolic_queries, text_queries, text_symbolic_query_idx, query_structure, idx):
        #if query_structure in query_only2query_structure.keys():
        #    alpha_embedding_symbolic, beta_embedding_symbolic, idx = self.embed_symbolic_query_beta(symbolic_queries, text_queries, text_symbolic_query_idx, query_only2query_structure[query_structure], idx)
        #    alpha_embedding_text, beta_embedding_text = self.embed_text_query_beta(text_queries)
        #    alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_symbolic, alpha_embedding_text]), torch.stack([beta_embedding_symbolic, beta_embedding_text]))
        #    return alpha_embedding, beta_embedding, idx
        #else:     
        alpha_embedding_symbolic, beta_embedding_symbolic, idx = self.embed_symbolic_query_beta(symbolic_queries, text_queries, text_symbolic_query_idx, query_structure, idx)
        alpha_embedding_text, beta_embedding_text = self.embed_text_query_beta(text_queries)
        alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_symbolic, alpha_embedding_text]), torch.stack([beta_embedding_symbolic, beta_embedding_text]))
        return alpha_embedding, beta_embedding, idx
    
    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_symbolic_queries_dict, batch_text_queries_dict, batch_text_symbolic_query_idx, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_2union_idxs, all_2union_alpha_embeddings, all_2union_beta_embeddings = [], [], []
        all_3union_idxs, all_3union_alpha_embeddings, all_3union_beta_embeddings = [], [], []
        for query_structure in batch_symbolic_queries_dict:
            if '2u' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_hybrid_query_beta(self.transform_union_query(batch_symbolic_queries_dict[query_structure], query_structure), 
                                                self.transform_union_text_query(batch_text_queries_dict[query_structure], query_structure),
                                                self.transform_union_text_symbolic_query_idx(batch_text_symbolic_query_idx[query_structure], query_structure),
                                                self.transform_union_structure(query_structure), 0)
                all_2union_idxs.extend(batch_idxs_dict[query_structure])
                all_2union_alpha_embeddings.append(alpha_embedding)
                all_2union_beta_embeddings.append(beta_embedding)
            elif '3u' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_hybrid_query_beta(self.transform_union_query(batch_symbolic_queries_dict[query_structure], query_structure), 
                                                self.transform_union_text_query(batch_text_queries_dict[query_structure], query_structure),
                                                self.transform_union_text_symbolic_query_idx(batch_text_symbolic_query_idx[query_structure], query_structure),
                                                self.transform_union_structure(query_structure), 0)
                all_3union_idxs.extend(batch_idxs_dict[query_structure])
                all_3union_alpha_embeddings.append(alpha_embedding)
                all_3union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_hybrid_query_beta(batch_symbolic_queries_dict[query_structure], 
                                                                           batch_text_queries_dict[query_structure], 
                                                                           batch_text_symbolic_query_idx[query_structure],
                                                                           query_structure, 
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)
        
        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
        if len(all_2union_alpha_embeddings) > 0:
            all_2union_alpha_embeddings = torch.cat(all_2union_alpha_embeddings, dim=0).unsqueeze(1)
            all_2union_beta_embeddings = torch.cat(all_2union_beta_embeddings, dim=0).unsqueeze(1)
            all_2union_alpha_embeddings = all_2union_alpha_embeddings.view(all_2union_alpha_embeddings.shape[0]//2, 2, 1, -1)
            all_2union_beta_embeddings = all_2union_beta_embeddings.view(all_2union_beta_embeddings.shape[0]//2, 2, 1, -1)
            all_2union_dists = torch.distributions.beta.Beta(all_2union_alpha_embeddings, all_2union_beta_embeddings)
        if len(all_3union_alpha_embeddings) > 0:
            all_3union_alpha_embeddings = torch.cat(all_3union_alpha_embeddings, dim=0).unsqueeze(1)
            all_3union_beta_embeddings = torch.cat(all_3union_beta_embeddings, dim=0).unsqueeze(1)
            all_3union_alpha_embeddings = all_3union_alpha_embeddings.view(all_3union_alpha_embeddings.shape[0]//3, 3, 1, -1)
            all_3union_beta_embeddings = all_3union_beta_embeddings.view(all_3union_beta_embeddings.shape[0]//3, 3, 1, -1)
            all_3union_dists = torch.distributions.beta.Beta(all_3union_alpha_embeddings, all_3union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_2union_idxs+all_3union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=positive_sample_regular)).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            if len(all_2union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_2union_idxs] # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=positive_sample_union)).unsqueeze(1).unsqueeze(1))
                positive_2union_logit = self.cal_logit_beta(positive_embedding, all_2union_dists)
                positive_2union_logit = torch.max(positive_2union_logit, dim=1)[0]
            else:
                positive_2union_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            if len(all_3union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_3union_idxs] # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=positive_sample_union)).unsqueeze(1).unsqueeze(1))
                positive_3union_logit = self.cal_logit_beta(positive_embedding, all_3union_dists)
                positive_3union_logit = torch.max(positive_3union_logit, dim=1)[0]
            else:
                positive_3union_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            positive_logit = torch.cat([positive_logit, positive_2union_logit, positive_3union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=negative_sample_regular.view(-1))).view(batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            if len(all_2union_alpha_embeddings) > 0:
                negative_sample_2union = negative_sample[all_2union_idxs]
                batch_size, negative_size = negative_sample_2union.shape
                negative_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=negative_sample_2union.view(-1))).view(batch_size, 1, negative_size, -1))
                negative_2union_logit = self.cal_logit_beta(negative_embedding, all_2union_dists)
                negative_2union_logit = torch.max(negative_2union_logit, dim=1)[0]
            else:
                negative_2union_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            if len(all_3union_alpha_embeddings) > 0:
                negative_sample_3union = negative_sample[all_3union_idxs]
                batch_size, negative_size = negative_sample_3union.shape
                negative_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=negative_sample_3union.view(-1))).view(batch_size, 1, negative_size, -1))
                negative_3union_logit = self.cal_logit_beta(negative_embedding, all_3union_dists)
                negative_3union_logit = torch.max(negative_3union_logit, dim=1)[0]
            else:
                negative_3union_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            negative_logit = torch.cat([negative_logit, negative_2union_logit, negative_3union_logit], dim=0)
        
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_2union_idxs+all_3union_idxs
    
    def embed_symbolic_query_gamma(self, symbolic_queries, text_queries, text_symbolic_query_idx, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                # Compute attention scores (dot-product attention)
                scores_text = torch.bmm(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0), torch.index_select(self.textual_query_embeddings, dim=0, index=text_queries).transpose(0, 1).unsqueeze(0))
                attn_weights_text = torch.softmax(scores_text, dim=-1)
                attended_Qtext = torch.bmm(attn_weights_text, torch.index_select(self.textual_query_embeddings, dim=0, index=text_queries).unsqueeze(0))
                
                scores_sym = torch.bmm(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0), torch.index_select(self.textual_symbolic_query_embeddings, dim=0, index=text_symbolic_query_idx).transpose(0, 1).unsqueeze(0))
                attn_weights_sym = torch.softmax(scores_sym, dim=-1)
                attended_Qsym = torch.bmm(attn_weights_sym, torch.index_select(self.textual_symbolic_query_embeddings, dim=0, index=text_symbolic_query_idx).unsqueeze(0))
                
                gate_text = torch.sigmoid(self.linear_gate_text(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0) + attended_Qtext))
                gate_sym = torch.sigmoid(self.linear_gate_sym(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0) + attended_Qsym))
                ent_bert_embedding_ca = torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]).unsqueeze(0) + gate_text * attended_Qtext + gate_sym * attended_Qsym

                embedding = self.entity_regularizer(self.ent_linear(ent_bert_embedding_ca.squeeze(0)))
                #embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=symbolic_queries[:, idx]))) 
                # self.bert(input_ids=ent_title, attention_mask=ent_title_m)[0][:, 0, :]
                
                alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1) 
                # self.bert(input_ids=ent_title, attention_mask=ent_title_m)[0][:, 0, :]
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_symbolic_query_gamma(symbolic_queries, text_queries, text_symbolic_query_idx, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (symbolic_queries[:, idx] == -2).all()
                    alpha_embedding = 1. / alpha_embedding
                    indicator_positive = beta_embedding >= 1
                    indicator_negative = beta_embedding < 1
                    beta_embedding[indicator_positive] = beta_embedding[indicator_positive] - 0.07
                    beta_embedding[indicator_negative] = beta_embedding[indicator_negative] + 0.07
                else:
                    r_embedding = self.rel_linear(torch.index_select(self.rel_bert_embeddings, dim=0, index=symbolic_queries[:, idx]))
                    alpha_r_embedding, beta_r_embedding = torch.chunk(r_embedding, 2, dim=-1)
                    alpha_embedding, beta_embedding = self.projection_net(alpha_embedding, beta_embedding,
                                                                          alpha_r_embedding, beta_r_embedding)
                idx += 1
        else:
            if self.is_u:
                alpha_embedding_list = []
                beta_embedding_list = []
                for i in range(len(query_structure)):
                    alpha_embedding, beta_embedding, idx = self.embed_symbolic_query_gamma(symbolic_queries, text_queries, text_symbolic_query_idx, query_structure[i], idx)
                    alpha_embedding_list.append(alpha_embedding)
                    beta_embedding_list.append(beta_embedding)

                alpha_embedding, beta_embedding = self.union_net(torch.stack(alpha_embedding_list),
                                                                 torch.stack(beta_embedding_list))
            else:
                alpha_embedding_list = []
                beta_embedding_list = []
                for i in range(len(query_structure)):
                    alpha_embedding, beta_embedding, idx = self.embed_symbolic_query_gamma(symbolic_queries, text_queries, text_symbolic_query_idx, query_structure[i], idx)
                    alpha_embedding_list.append(alpha_embedding)
                    beta_embedding_list.append(beta_embedding)
                alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                                  torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def embed_text_query_gamma(self, text_queries):
        embedding = self.ent_linear(torch.index_select(self.textual_query_embeddings, dim=0, index=text_queries))
        #embedding = self.entity_regularizer(self.ent_linear(self.bert(input_ids=text_queries[:,0,:], attention_mask=text_queries[:,1,:])[0][:, 0, :]))
        alpha_embedding, beta_embedding = torch.chunk(self.entity_regularizer(embedding), 2, dim=-1)

        return alpha_embedding, beta_embedding                                          

    def embed_hybrid_query_gamma(self, symbolic_queries, text_queries, text_symbolic_query_idx, query_structure, idx):
        alpha_embedding_symbolic, beta_embedding_symbolic, idx = self.embed_symbolic_query_gamma(symbolic_queries, text_queries, text_symbolic_query_idx, query_structure, idx)
        alpha_embedding_text, beta_embedding_text = self.embed_text_query_gamma(text_queries)
        alpha_embedding, beta_embedding = self.center_net(torch.stack([alpha_embedding_symbolic, alpha_embedding_text]), torch.stack([beta_embedding_symbolic, beta_embedding_text]))
        return alpha_embedding, beta_embedding, idx
    
    def cal_logit_gamma(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.gamma.Gamma(alpha_embedding, beta_embedding)
        distance = torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        logit = self.gamma - distance

        return logit

    def forward_gamma(self, positive_sample, negative_sample, subsampling_weight, batch_symbolic_queries_dict, batch_text_queries_dict, batch_text_symbolic_query_idx, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_2union_idxs, all_2union_alpha_embeddings, all_2union_beta_embeddings = [], [], []
        all_3union_idxs, all_3union_alpha_embeddings, all_3union_beta_embeddings = [], [], []
        for query_structure in batch_symbolic_queries_dict:
            if '2u' in self.query_name_dict[query_structure]:
                self.is_u = True
                alpha_embedding, beta_embedding, _ = \
                    self.embed_hybrid_query_gamma(self.transform_gamma_union_query(batch_symbolic_queries_dict[query_structure], query_structure), 
                                                batch_text_queries_dict[query_structure],
                                                batch_text_symbolic_query_idx[query_structure],
                                                self.transform_gamma_union_structure(query_structure), 0)
                all_2union_idxs.extend(batch_idxs_dict[query_structure])
                all_2union_alpha_embeddings.append(alpha_embedding)
                all_2union_beta_embeddings.append(beta_embedding)
            elif '3u' in self.query_name_dict[query_structure]:
                self.is_u = True
                alpha_embedding, beta_embedding, _ = \
                    self.embed_hybrid_query_gamma(self.transform_gamma_union_query(batch_symbolic_queries_dict[query_structure], query_structure), 
                                                batch_text_queries_dict[query_structure],
                                                batch_text_symbolic_query_idx[query_structure],
                                                self.transform_gamma_union_structure(query_structure), 0)
                all_3union_idxs.extend(batch_idxs_dict[query_structure])
                all_3union_alpha_embeddings.append(alpha_embedding)
                all_3union_beta_embeddings.append(beta_embedding)
            else:
                self.is_u = False
                alpha_embedding, beta_embedding, _ = self.embed_hybrid_query_gamma(batch_symbolic_queries_dict[query_structure], 
                                                                           batch_text_queries_dict[query_structure], 
                                                                           batch_text_symbolic_query_idx[query_structure],
                                                                           query_structure, 
                                                                           0)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)
        
        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.gamma.Gamma(all_alpha_embeddings, all_beta_embeddings)
        if len(all_2union_alpha_embeddings) > 0:
            all_2union_alpha_embeddings = torch.cat(all_2union_alpha_embeddings, dim=0).unsqueeze(1)
            all_2union_beta_embeddings = torch.cat(all_2union_beta_embeddings, dim=0).unsqueeze(1)
            all_2union_dists = torch.distributions.gamma.Gamma(all_2union_alpha_embeddings, all_2union_beta_embeddings)
        if len(all_3union_alpha_embeddings) > 0:
            all_3union_alpha_embeddings = torch.cat(all_3union_alpha_embeddings, dim=0).unsqueeze(1)
            all_3union_beta_embeddings = torch.cat(all_3union_beta_embeddings, dim=0).unsqueeze(1)
            all_3union_dists = torch.distributions.gamma.Gamma(all_3union_alpha_embeddings, all_3union_beta_embeddings)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_2union_idxs+all_3union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs] # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=positive_sample_regular)).unsqueeze(1))
                positive_logit = self.cal_logit_gamma(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            if len(all_2union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_2union_idxs] # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=positive_sample_union)).unsqueeze(1))
                positive_2union_logit = self.cal_logit_gamma(positive_embedding, all_2union_dists)
                #positive_2union_logit = torch.max(positive_2union_logit, dim=1)[0]
            else:
                positive_2union_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            if len(all_3union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_3union_idxs] # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=positive_sample_union)).unsqueeze(1))
                positive_3union_logit = self.cal_logit_gamma(positive_embedding, all_3union_dists)
                #positive_3union_logit = torch.max(positive_3union_logit, dim=1)[0]
            else:
                positive_3union_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            positive_logit = torch.cat([positive_logit, positive_2union_logit, positive_3union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=negative_sample_regular.view(-1))).view(batch_size, negative_size, -1))
                negative_logit = self.cal_logit_gamma(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            if len(all_2union_alpha_embeddings) > 0:
                negative_sample_2union = negative_sample[all_2union_idxs]
                batch_size, negative_size = negative_sample_2union.shape
                negative_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=negative_sample_2union.view(-1))).view(batch_size, negative_size, -1))
                negative_2union_logit = self.cal_logit_gamma(negative_embedding, all_2union_dists)
                #negative_2union_logit = torch.max(negative_2union_logit, dim=1)[0]
            else:
                negative_2union_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)

            if len(all_3union_alpha_embeddings) > 0:
                negative_sample_3union = negative_sample[all_3union_idxs]
                batch_size, negative_size = negative_sample_3union.shape
                negative_embedding = self.entity_regularizer(self.ent_linear(torch.index_select(self.ent_bert_embeddings, dim=0, index=negative_sample_3union.view(-1))).view(batch_size, negative_size, -1))
                
                negative_3union_logit = self.cal_logit_gamma(negative_embedding, all_3union_dists)
                #negative_3union_logit = torch.max(negative_3union_logit, dim=1)[0]
            else:
                negative_3union_logit = torch.Tensor([]).to(self.ent_bert_embeddings.device)
            negative_logit = torch.cat([negative_logit, negative_2union_logit, negative_3union_logit], dim=0)
        
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_2union_idxs+all_3union_idxs
 

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_symbolic_queries, batch_textual_queries, text_symbolic_query_idx, query_structures = next(train_iterator)
        batch_symbolic_queries_dict = collections.defaultdict(list)
        batch_text_queries_dict = collections.defaultdict(list)
        batch_text_symbolic_query_idx = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_symbolic_queries): # group queries with same structure
            batch_symbolic_queries_dict[query_structures[i]].append(query)
            batch_text_queries_dict[query_structures[i]].append(batch_textual_queries[i])
            batch_text_symbolic_query_idx[query_structures[i]].append(text_symbolic_query_idx[i])
            #batch_text_queries_dict[query_structures[i]].append([batch_textual_queries['input_ids'][i,:].tolist(), batch_textual_queries['attention_mask'][i,:].tolist()])
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_symbolic_queries_dict:
            if args.cuda:
                batch_symbolic_queries_dict[query_structure] = torch.LongTensor(batch_symbolic_queries_dict[query_structure]).cuda()
                batch_text_queries_dict[query_structure] = torch.LongTensor(batch_text_queries_dict[query_structure]).cuda()
                batch_text_symbolic_query_idx[query_structure] = torch.LongTensor(batch_text_symbolic_query_idx[query_structure]).cuda()
            else:
                batch_symbolic_queries_dict[query_structure] = torch.LongTensor(batch_symbolic_queries_dict[query_structure])
                batch_text_queries_dict[query_structure] = torch.LongTensor(batch_text_queries_dict[query_structure])
                batch_text_symbolic_query_idx[query_structure] = torch.LongTensor(batch_text_symbolic_query_idx[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_symbolic_queries_dict, batch_text_queries_dict, batch_text_symbolic_query_idx, batch_idxs_dict)
        
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, id2text_queries, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries_unflatten, batch_text_symbolic_query_idx, batch_symbolic_queries, batch_textual_queries, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_symbolic_queries_dict = collections.defaultdict(list)
                batch_text_queries_dict = collections.defaultdict(list)
                batch_text_symbolic_query_idx_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(batch_symbolic_queries): # group queries with same structure
                    batch_symbolic_queries_dict[query_structures[i]].append(query)
                    batch_text_queries_dict[query_structures[i]].append(batch_textual_queries[i])
                    batch_text_symbolic_query_idx_dict[query_structures[i]].append(batch_text_symbolic_query_idx[i])
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_symbolic_queries_dict:
                    if args.cuda:
                        batch_symbolic_queries_dict[query_structure] = torch.LongTensor(batch_symbolic_queries_dict[query_structure]).cuda()
                        batch_text_queries_dict[query_structure] = torch.LongTensor(batch_text_queries_dict[query_structure]).cuda()
                        batch_text_symbolic_query_idx_dict[query_structure] = torch.LongTensor(batch_text_symbolic_query_idx_dict[query_structure]).cuda()
                    else:
                        batch_symbolic_queries_dict[query_structure] = torch.LongTensor(batch_symbolic_queries_dict[query_structure])
                        batch_text_queries_dict[query_structure] = torch.LongTensor(batch_text_queries_dict[query_structure])
                        batch_text_symbolic_query_idx_dict[query_structure] = torch.LongTensor(batch_text_symbolic_query_idx_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda() 

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_symbolic_queries_dict, batch_text_queries_dict, batch_text_symbolic_query_idx_dict, batch_idxs_dict)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[(query[0], id2text_queries[query[1]])]
                    easy_answer = easy_answers[(query[0], id2text_queries[query[1]])]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics