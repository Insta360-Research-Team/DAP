from typing import Callable, Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import *
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim: Optional[int],
    ori_attn_proc: object,
) -> object:
    return ori_attn_proc

def set_flux_transformer_attn_processor(
    transformer: FluxTransformer2DModel,
    set_attn_proc_func: Callable = default_set_attn_proc_func,
    set_attn_module_names: Optional[list[str]] = None,
) -> None:
    do_set_processor = lambda name, module_names: (
        any([name.startswith(module_name) for module_name in module_names])
        if module_names is not None
        else True
    )  # prefix match

    attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        dim_head = transformer.config.attention_head_dim
        num_heads = transformer.config.num_attention_heads
        if name.endswith("attn.processor"):
            attn_procs[name] = (
                set_attn_proc_func(name, dim_head, num_heads, attn_processor)
                if do_set_processor(name, set_attn_module_names)
                else attn_processor
            )

    transformer.set_attn_processor(attn_procs)

class PersonalizeAnythingAttnProcessor:
    
    def __init__(self, name, mask, device, tau=0.98, concept_process=False, shift_mask = None, img_dims=4096):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.name = name
        self.mask = mask.view(img_dims).bool().to(device)
        self.device = device
        self.tau = tau
        self.concept_process = concept_process
        self.img_dims = img_dims
        
        if shift_mask is None:
            self.shift_mask = self.mask
        else:
            self.shift_mask = shift_mask.view(img_dims).bool().to(device)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        ###################################################################################
        if timestep is not None:
            timestep = timestep
            
        concept_process = self.concept_process # token concatenation
        c_q = concept_process and True # if token concatenation is applied to q
        c_kv = concept_process and True  # if token concatenation is applied to kv
        
        t_flag = timestep > self.tau  # token replacement
        r_q = True and t_flag # if token concatenation is applied to q
        r_k = True and t_flag # if token concatenation is applied to k
        r_v = True and t_flag # if token concatenation is applied to v    
        
        if encoder_hidden_states is not None:
            concept_feature_ = hidden_states[0, self.mask, :] 
        else:
            concept_feature_ = hidden_states[0, 512:, :][self.mask, :] 
        
        if r_k or r_q or r_v:
            r_hidden_states = hidden_states
            if encoder_hidden_states is not None:
                r_hidden_states[1, self.shift_mask, :] = concept_feature_
            else:
                text_hidden_states = hidden_states[1, :512, :]
                image_hidden_states = hidden_states[1, 512:, :]
                image_hidden_states[self.shift_mask, :] = concept_feature_
                
                r_hidden_states[1] = torch.cat([text_hidden_states, image_hidden_states], dim=0)
        ###################################################################################

        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = attn.to_q(hidden_states)
        
        ###################################################################################        
        if r_k:
            key = attn.to_k(r_hidden_states)
        if r_q:
            query = attn.to_q(r_hidden_states)
        if r_v:
            value = attn.to_v(r_hidden_states)

        if concept_process:    
            if c_q:
                c_query = attn.to_q(concept_feature_)
                c_query = c_query.repeat(query.shape[0], 1, 1)
                query = torch.cat([query, c_query], dim=1)
            if c_kv:
                c_key = attn.to_k(concept_feature_)
                c_key = c_key.repeat(key.shape[0], 1, 1)
                c_value = attn.to_v(concept_feature_)
                c_value = c_value.repeat(value.shape[0], 1, 1)
                key = torch.cat([key, c_key], dim=1)
                value = torch.cat([value, c_value], dim=1)
        ###################################################################################

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            
            # use original position emb or text emb
            if not c_q:
                query = apply_rotary_emb(query, image_rotary_emb)
            if not c_kv:
                key = apply_rotary_emb(key, image_rotary_emb)
            
            ###################################################################################
            # get original position emb
            def get_concept_rotary_emb(ori_rotary_emb, mask):
                enc_emb = ori_rotary_emb[:512, :]
                hid_emb = ori_rotary_emb[512:, :]
                concept_emb = hid_emb[mask, :]
                
                image_rotary_emb = torch.cat([enc_emb, hid_emb, concept_emb], dim=0)
                return image_rotary_emb
                
            if concept_process:
                # 1. use original position emb
                image_rotary_emb_0 = get_concept_rotary_emb(image_rotary_emb[0], self.shift_mask)
                image_rotary_emb_1 = get_concept_rotary_emb(image_rotary_emb[1], self.shift_mask)
                image_rotary_emb = (image_rotary_emb_0, image_rotary_emb_1)
                
                # 2. use text emb
                # dims = (self.mask == 1).sum().item()
                # concept_rotary_emb_0 = torch.ones((dims, 128)).to(self.device)
                # concept_rotary_emb_1 = torch.zeros((dims, 128)).to(self.device)
                # image_rotary_emb = (
                #     torch.cat([image_rotary_emb[0], concept_rotary_emb_0], dim=0), 
                #     torch.cat([image_rotary_emb[1], concept_rotary_emb_1], dim=0))      
                
                if c_q:
                    query = apply_rotary_emb(query, image_rotary_emb)
                if c_kv:
                    key = apply_rotary_emb(key, image_rotary_emb)
            ###################################################################################

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)    
        
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            ################################################################
            # restore after token concatenation
            hidden_states = hidden_states[:, :self.img_dims, :]
            ################################################################

            return hidden_states, encoder_hidden_states
        else:
            ################################################################
            dims = self.img_dims + 512
            hidden_states = hidden_states[:, :dims, :]
            ################################################################
            
            return hidden_states
        
class MultiPersonalizeAnythingAttnProcessor:
    
    def __init__(self, name, masks, device, tau=0.98, concept_process=False, shift_masks = None, img_dims=4096):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.name = name
        self.device = device
        self.tau = tau
        self.concept_process = concept_process
        self.img_dims = img_dims
        
        for i in range(len(masks)):
            masks[i] = masks[i].view(img_dims).bool().to(device)
        self.masks = masks
        
        if shift_masks is None:
            self.shift_masks = self.masks
        else:
            for i in range(len(shift_masks)):
                shift_masks[i] = shift_masks[i].view(img_dims).bool().to(device)
            self.shift_masks = shift_masks

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        ###################################################################################
        if timestep is not None:
            timestep = timestep
            
        concept_process = self.concept_process # token concatenation
        c_q = concept_process and True # if token concatenation is applied to q
        c_kv = concept_process and True  # if token concatenation is applied to kv
        
        t_flag = timestep > self.tau  # token replacement
        r_q = True and t_flag # if token concatenation is applied to q
        r_k = True and t_flag # if token concatenation is applied to k
        r_v = True and t_flag # if token concatenation is applied to v    
        
        concept_features = []
        r_hidden_states = hidden_states
        for id, mask in enumerate(self.masks):
            if encoder_hidden_states is not None:
                concept_feature_ = hidden_states[id, mask, :] 
            else:
                concept_feature_ = hidden_states[id, 512:, :][mask, :]
                
            shift_mask = self.shift_masks[id]
            concept_features.append(concept_feature_)
            
            if r_k or r_q or r_v:
                if encoder_hidden_states is not None:
                    r_hidden_states[-1, shift_mask, :] = concept_feature_
                else:
                    text_hidden_states = r_hidden_states[-1, :512, :]
                    image_hidden_states = r_hidden_states[-1, 512:, :]
                    image_hidden_states[shift_mask, :] = concept_feature_
                    r_hidden_states[-1] = torch.cat([text_hidden_states, image_hidden_states], dim=0)
        ###################################################################################

        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = attn.to_q(hidden_states)
        
        ###################################################################################        
        if r_k:
            key = attn.to_k(r_hidden_states)
        if r_q:
            query = attn.to_q(r_hidden_states)
        if r_v:
            value = attn.to_v(r_hidden_states)

        if concept_process:  
            for concept_feature_ in concept_features:  
                if c_q:
                    c_query = attn.to_q(concept_feature_)
                    c_query = c_query.repeat(query.shape[0], 1, 1)
                    query = torch.cat([query, c_query], dim=1)
                if c_kv:
                    c_key = attn.to_k(concept_feature_)
                    c_key = c_key.repeat(key.shape[0], 1, 1)
                    
                    c_value = attn.to_v(concept_feature_)
                    c_value = c_value.repeat(value.shape[0], 1, 1)
                    
                    key = torch.cat([key, c_key], dim=1)
                    value = torch.cat([value, c_value], dim=1)
        ###################################################################################

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            
            # use original position emb or text emb
            if not c_q:
                query = apply_rotary_emb(query, image_rotary_emb)
            if not c_kv:
                key = apply_rotary_emb(key, image_rotary_emb)
            
            ###################################################################################
            def get_concept_rotary_emb(ori_rotary_emb, shift_masks):
                enc_emb = ori_rotary_emb[:512, :]
                hid_emb = ori_rotary_emb[512:, :]
                
                concept_embs = []
                for mask in shift_masks:
                    concept_embs.append(hid_emb[mask, :])
                concept_emb = torch.cat(concept_embs, dim=0) if len(concept_embs) > 0 else torch.zeros(0, hid_emb.shape[1], device=hid_emb.device)
                image_rotary_emb = torch.cat([enc_emb, hid_emb, concept_emb], dim=0)
                return image_rotary_emb
            
            if concept_process:
                # 1. use original position emb with plural masks
                image_rotary_emb_0 = get_concept_rotary_emb(image_rotary_emb[0], self.shift_masks)
                image_rotary_emb_1 = get_concept_rotary_emb(image_rotary_emb[1], self.shift_masks)
                image_rotary_emb = (image_rotary_emb_0, image_rotary_emb_1)
                
                # 2. use text emb with plural masks
                # total_dims = sum((mask == 1).sum().item() for mask in self.masks)
                # concept_rotary_emb_0 = torch.ones((total_dims, 128)).to(self.device)
                # concept_rotary_emb_1 = torch.zeros((total_dims, 128)).to(self.device)
                # image_rotary_emb = (
                #     torch.cat([image_rotary_emb[0], concept_rotary_emb_0], dim=0), 
                #     torch.cat([image_rotary_emb[1], concept_rotary_emb_1], dim=0)
                # )
                
                if c_q:
                    query = apply_rotary_emb(query, image_rotary_emb)
                if c_kv:
                    key = apply_rotary_emb(key, image_rotary_emb)
            ###################################################################################

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)    
        
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            ################################################################
            # restore after token concatenation
            hidden_states = hidden_states[:, :self.img_dims, :]
            ################################################################

            return hidden_states, encoder_hidden_states
        else:
            ################################################################
            dims = self.img_dims + 512
            hidden_states = hidden_states[:, :dims, :]
            ################################################################
            
            return hidden_states