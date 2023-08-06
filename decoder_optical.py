import torch
import torch.nn as nn


# two layers of GRU
class Gru_cond_layer(nn.Module):
    def __init__(self, params):
        super(Gru_cond_layer, self).__init__()
        # attention
        self.conv_Ua_off = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.conv_Ua_on = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.fc_Wa_off = nn.Linear(params['n'], params['dim_attention'], bias=False)
        # self.fc_Wa_on = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_Q_off = nn.Conv2d(1, 512, kernel_size=11, bias=False, padding=5)
        self.conv_Q_on = nn.Conv2d(1, 512, kernel_size=11, bias=False, padding=5)
        self.fc_Uf_off = nn.Linear(512, params['dim_attention'])
        self.fc_Uf_on = nn.Linear(512, params['dim_attention'])

        self.fc_va_off = nn.Linear(params['dim_attention'], 1)
        self.fc_va_on  = nn.Linear(params['dim_attention'], 1)
        
        self.fuse_ct = nn.Linear(params['D']*2, params['D'])

        # the first GRU layer
        self.fc_Wyz = nn.Linear(params['m'], params['n'])
        self.fc_Wyr = nn.Linear(params['m'], params['n'])
        self.fc_Wyh = nn.Linear(params['m'], params['n'])

        self.fc_Uhz = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhr = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhh = nn.Linear(params['n'], params['n'], bias=False)

        # the second GRU layer
        self.fc_Wcz = nn.Linear(params['D'], params['n'], bias=False)
        self.fc_Wcr = nn.Linear(params['D'], params['n'], bias=False)
        self.fc_Wch = nn.Linear(params['D'], params['n'], bias=False)

        self.fc_Uhz2 = nn.Linear(params['n'], params['n'])
        self.fc_Uhr2 = nn.Linear(params['n'], params['n'])
        self.fc_Uhh2 = nn.Linear(params['n'], params['n'])

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, params, embedding, mask=None, context_off=None, context_on=None, context_mask=None, one_step=False, init_state=None,
                alpha_past_off=None, alpha_past_on=None):
        n_steps = embedding.shape[0]
        n_samples = embedding.shape[1]

        Ua_ctx_off = self.conv_Ua_off(context_off)
        Ua_ctx_off = Ua_ctx_off.permute(2, 3, 0, 1) 

        Ua_ctx_on = self.conv_Ua_on(context_on)
        Ua_ctx_on = Ua_ctx_on.permute(2, 3, 0, 1) 

        state_below_z = self.fc_Wyz(embedding)
        state_below_r = self.fc_Wyr(embedding)
        state_below_h = self.fc_Wyh(embedding)

        if one_step:
            if mask is None:
                mask = torch.ones(embedding.shape[0]).cuda()
            h2ts, cts, alphas_off, alphas_on, alpha_pasts_off, alpha_pasts_on = self._step_slice(mask, state_below_r, state_below_z, state_below_h,
                                                              init_state, context_off, context_on, context_mask, alpha_past_off, alpha_past_on, Ua_ctx_off, Ua_ctx_on)
        else:
            alpha_past_off = torch.zeros(n_samples, context_off.shape[2], context_off.shape[3]).cuda()
            alpha_past_on = torch.zeros(n_samples, context_on.shape[2], context_on.shape[3]).cuda()
            h2t = init_state
            h2ts = torch.zeros(n_steps, n_samples, params['n']).cuda()
            cts = torch.zeros(n_steps, n_samples, params['D']).cuda()
            alphas_off = (torch.zeros(n_steps, n_samples, context_off.shape[2], context_off.shape[3])).cuda()
            alphas_on = (torch.zeros(n_steps, n_samples, context_on.shape[2], context_on.shape[3])).cuda()
            alpha_pasts_off = torch.zeros(n_steps, n_samples, context_off.shape[2], context_off.shape[3]).cuda()
            alpha_pasts_on = torch.zeros(n_steps, n_samples, context_on.shape[2], context_on.shape[3]).cuda()
            for i in range(n_steps):
                h2t, ct, alpha_off, alpha_on, alpha_past_off, alpha_past_on = self._step_slice(mask[i], state_below_r[i], state_below_z[i],
                                                              state_below_h[i], h2t, context_off, context_on, context_mask, alpha_past_off, alpha_past_on,
                                                              Ua_ctx_off, Ua_ctx_on)
                h2ts[i] = h2t
                cts[i] = ct
                alphas_off[i] = alpha_off
                alphas_on[i] = alpha_on
                alpha_pasts_off[i] = alpha_past_off
                alpha_pasts_on[i] = alpha_past_on
        return h2ts, cts, alphas_off, alphas_on, alpha_pasts_off, alpha_pasts_on

    # one step of two GRU layers
    def _step_slice(self, mask, state_below_r, state_below_z, state_below_h, h, ctx_off, ctx_on, ctx_mask, alpha_past_off, alpha_past_on, Ua_ctx_off, Ua_ctx_on ):
        # the first GRU layer
        z1 = torch.sigmoid(self.fc_Uhz(h) + state_below_z)
        r1 = torch.sigmoid(self.fc_Uhr(h) + state_below_r)
        h1_p = torch.tanh(self.fc_Uhh(h) * r1 + state_below_h)
        h1 = z1 * h + (1. - z1) * h1_p
        h1 = mask[:, None] * h1 + (1. - mask)[:, None] * h

        # attention offline
        Wa_h1 = self.fc_Wa_off(h1)
        alpha_past_ = alpha_past_off[:, None, :, :]
        cover_F = self.conv_Q_off(alpha_past_).permute(2, 3, 0, 1)
        cover_vector_off = self.fc_Uf_off(cover_F)

        alpha_past_ = alpha_past_on[:, None, :, :]
        cover_F = self.conv_Q_on(alpha_past_).permute(2, 3, 0, 1)
        cover_vector_on = self.fc_Uf_on(cover_F)

        attention_score_off = torch.tanh(Ua_ctx_off + Wa_h1[None, None, :, :] + cover_vector_off + cover_vector_on)
        alpha_off = self.fc_va_off(attention_score_off)
        alpha_off = alpha_off.view(alpha_off.shape[0], alpha_off.shape[1], alpha_off.shape[2])
        alpha_off = alpha_off - alpha_off.max()
        alpha_off = torch.exp(alpha_off)
        if (ctx_mask is not None):
            alpha_off = alpha_off * ctx_mask.permute(1, 2, 0)

        alpha_off = alpha_off / (alpha_off.sum(1).sum(0)[None, None, :] + 1e-10)
        alpha_past_off = alpha_past_off + alpha_off.permute(2, 0, 1)


        # attention online
        # Wa_h1 = self.fc_Wa_on(h1)
        
        attention_score_on = torch.tanh(Ua_ctx_on + Wa_h1[None, None, :, :] + cover_vector_off + cover_vector_on)
        alpha_on = self.fc_va_on(attention_score_on)
        alpha_on = alpha_on.view(alpha_on.shape[0], alpha_on.shape[1], alpha_on.shape[2])
        alpha_on = alpha_on - alpha_on.max()
        alpha_on = torch.exp(alpha_on)

        if (ctx_mask is not None):
            alpha_on = alpha_on * ctx_mask.permute(1, 2, 0)
        alpha_on = alpha_on / (alpha_on.sum(1).sum(0)[None, None, :] + 1e-10)
        alpha_past_on = alpha_past_on + alpha_on.permute(2, 0, 1)

        # extract off context
        ct_off = (ctx_off * alpha_off.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)

        # extract on context
        ct_on = (ctx_on * alpha_on.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)

        ct = torch.concat((ct_off, ct_on) , dim = -1)
        ct = self.fuse_ct(ct)
        ct = self.dropout(ct)

        # the second GRU layer
        z2 = torch.sigmoid(self.fc_Wcz(ct) + self.fc_Uhz2(h1))
        r2 = torch.sigmoid(self.fc_Wcr(ct) + self.fc_Uhr2(h1))
        h2_p = torch.tanh(self.fc_Wch(ct) + self.fc_Uhh2(h1) * r2)
        h2 = z2 * h1 + (1. - z2) * h2_p
        h2 = mask[:, None] * h2 + (1. - mask)[:, None] * h1
        return h2, ct, alpha_off.permute(2, 0, 1),alpha_on.permute(2, 0, 1), alpha_past_off, alpha_past_on

# calculate probabilities
class Gru_prob(nn.Module):
    def __init__(self, params):
        super(Gru_prob, self).__init__()
        self.fc_Wct = nn.Linear(params['D'], params['m'])
        self.fc_Wht = nn.Linear(params['n'], params['m'])
        self.fc_Wyt = nn.Linear(params['m'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_W0 = nn.Linear(int(params['m'] / 2), params['K'])

    def forward(self, cts, hts, emb, use_dropout):
        logit = self.fc_Wct(cts) + self.fc_Wht(hts) + self.fc_Wyt(emb)

        # maxout
        shape = logit.shape
        shape2 = int(shape[2] / 2)
        shape3 = 2
        logit = logit.view(shape[0], shape[1], shape2, shape3)
        logit = logit.max(3)[0]

        if use_dropout:
            logit = self.dropout(logit)

        out = self.fc_W0(logit)
        return out
