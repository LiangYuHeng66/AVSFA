from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn
# from model.ot import optimal_transport_dist
# from model.token import VTC_Module


class AVSFA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'VTC' in args.loss_names:
            self.VTC = VTC_Module(ratio=args.img_k_ratio)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x,"text")
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        image_feats,att_score = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()

    def encode_text(self, text):
        x,att_score = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_token(self, image):
        image_feats,att_score = self.base_model.encode_image(image)
        _,i_tse_f = self.visul_emb_layer(image_feats, att_score)
        return i_tse_f.float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        with torch.autocast(dtype=torch.float16, device_type='cuda'):
            image_feats, image_att_scores, text_feats, text_att_scores = self.base_model(images, caption_ids)

        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        if 'VTC' in self.current_task:
            _,i_feats_fg = self.VTC(image_feats, image_att_scores)
            i_feats_new = i_feats + self.args.i_feats_fg_gamma * i_feats_fg
        else:
            i_feats_new = i_feats

        if 'FGSM' in self.current_task:
            ot_loss = optimal_transport_dist(text_feats.float(), image_feats.float())
            ot_loss = ot_loss * self.args.ot_gamma
            ret.update({'ot_loss': ot_loss})

        if 'TAL' in self.current_task:
            ret.update({'TAL_loss': objectives.compute_TAL(i_feats_new, t_feats, batch['pids'], self.args.tau, self.args.margin)})


        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats_new.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})


        return ret


def build_finetune_model(args, num_classes=11003):
    model = AVSFA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
