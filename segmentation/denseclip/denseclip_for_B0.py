# denseclip_for_B0.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

from .untils import tokenize

import os


@SEGMENTORS.register_module()
class DenseCLIP_B0(BaseSegmentor):
    
    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 decode_head,
                 class_names,
                 context_length,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 neck=None,
                 tau=0.07,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 token_embed_dim=512, text_dim=1024,
                # ==== added code: B0 전용 옵션 ====
                 freeze_teacher=True,     # <-- B0에서는 True (기본)
                 aux_backbone=None,       # MaskCLIP+ 학생 경로(선택)
                 aux_aspp=None,
                 student_head_aspp=None,
                 **args):
        super(DenseCLIP_B0, self).__init__(init_cfg) # class명 DenseCLIP_B0으로 수정
        
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        # ---- Teacher 모듈 구성 (DenseCLIP Original) ----
        self.backbone = builder.build_backbone(backbone)
        self.text_encoder = builder.build_backbone(text_encoder)
        self.context_decoder = builder.build_backbone(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)
       
        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        assert self.with_decode_head
        
        # ---- added code: (선택) 학생 경로: MaskCLIP+ 노란 블록 ----
        self.aux_backbone = builder.build_backbone(aux_backbone) if aux_backbone else None
        self.aux_aspp = builder.build_head(aux_aspp) if aux_aspp else None
        self.student_head_aspp = builder.build_head(student_head_aspp) if student_head_aspp else None
        if self.student_head_aspp is not None:
            # teacher score(K개)를 학생 decoder 입력 채널(256)로 정렬
            self.score_adapter = nn.Sequential(
                nn.Conv2d(self.num_classes, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        # added code: 학생 경로 사용 여부 플래그
        self.use_student_head = (self.aux_backbone is not None
                                 and self.aux_aspp is not None
                                 and self.student_head_aspp is not None)

        # ---- added code: B0: Teacher 완전 동결 ----
        if freeze_teacher:
            self._freeze_teacher_modules()
    
        # denseclip_for_B0.py - __init__ 마지막에 추가
        print("=== Freeze check at init ===")

        TEACHER_PREFIXES = {'backbone', 'text_encoder', 'context_decoder', 'neck', 'decode_head'}
        STUDENT_PREFIXES = {'aux_backbone', 'aux_aspp', 'student_head_aspp', 'score_adapter'}

        for n, p in self.named_parameters():
            prefix = n.split('.', 1)[0]  # 맨 앞 모듈 속성명만 추출

            # 학생 먼저 체크 (elif로 중복방지)
            if prefix in STUDENT_PREFIXES:
                print(f"[Student] {n:40} trainable={p.requires_grad}")
                assert p.requires_grad, f"Student frozen by mistake: {n}"

            # teacher 체크
            elif prefix in TEACHER_PREFIXES or n.startswith('contexts') or n.startswith('gamma'):
                print(f"[Teacher] {n:40} trainable={p.requires_grad}")
                assert not p.requires_grad, f"Teacher not frozen: {n}"

            # 그 외(예: optimizer가 관리 안 하는 잡다 모듈)은 스킵


    # ----------------- added code: Freeze helpers -----------------
    def _freeze(self, m: nn.Module):
        if m is None: 
            return
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_teacher_modules(self):
        self._freeze(self.backbone)
        self._freeze(self.text_encoder)
        self._freeze(self.context_decoder)
        self._freeze(getattr(self, 'neck', None))
        self._freeze(self.decode_head)
        # contexts / gamma 파라미터도 고정
        self.contexts.requires_grad_(False)
        self.gamma.requires_grad_(False)
    # ---------------------------------------------------

    # ---- mmseg 기본 head 초기화들 ----
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)
    
    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)
            
    # ---- Teacher forward parts (동결되어도 forward는 사용) ----
    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    # ---- Teacher 후처리: text emb, score map 생성 + FPN입력 concat ----
    def after_extract_feat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat(
                [global_feat.reshape(B, C, 1), 
                 visual_embeddings.reshape(B, C, H*W)], dim=2
            ).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape

        # pixel-text score maps K
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
    
        # score concat to FPN input
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map
    
    # ----------------- Train / Test -----------------
    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _identity_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_aux, 'aux_identity'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """
        B0-zero: gt_semantic_seg == pseudo labels (.npz에서 로드),
        Teacher는 frozen 상태로 forward만; loss는 pseudo에 대해 계산.
        """
        
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        """
        B0-zero: gt_semantic_seg == pseudo labels (.npz에서 로드),
        Teacher는 frozen 상태로 forward만; loss는 pseudo에 대해 계산.
        """
        
        # ---- added code: Teacher forward (frozen) ----
        x = self.extract_feat(img)
        text_embeddings, x_orig, score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        losses = dict()

        # 기존 코드
        # losses = dict()
        # if self.text_head:
        #     x = [text_embeddings,] + x_orig
        # else:
        #     x = x_orig

        # loss_decode = self._decode_head_forward_train(x, img_metas,
        #                                               gt_semantic_seg)
        # losses.update(loss_decode)

        # if self.with_identity_head:
        #     loss_identity = self._identity_head_forward_train(
        #         score_map/self.tau, img_metas, gt_semantic_seg)
        #     losses.update(loss_identity)

        # if self.with_auxiliary_head:
        #     loss_aux = self._auxiliary_head_forward_train(
        #         _x_orig, img_metas, gt_semantic_seg)
        #     losses.update(loss_aux)

        # return losses

        # 수정 코드(0821 기준)
        # ★ teacher 디코더 입력 묶기
        # x_in = [text_embeddings] + x_orig if self.text_head else x_orig
        
        # losses = dict()
        
        # # added code
        # # 1) 메인(Teacher decode head)도 pseudo에 대해 감독
        # losses.update(self._decode_head_forward_train(x_in, img_metas, gt_semantic_seg))

        # # (선택) identity/aux head
        # if self.with_identity_head:
        #     losses.update(self._identity_head_forward_train(score_map / self.tau, img_metas, gt_semantic_seg))
        # if self.with_auxiliary_head:
        #     losses.update(self._auxiliary_head_forward_train(_x_orig, img_metas, gt_semantic_seg))

        # # 2) (선택) 학생 경로: dilated ResNet + ASPP + 별도 decoder
        # if (self.aux_backbone is not None) and (self.aux_aspp is not None) and (self.student_head_aspp is not None):
        #     feats_r50d = self.aux_backbone(img)                      # tuple of 4
        #     V = self.aux_aspp.forward_module(feats_r50d)             # (B,256,h,w)
            
        #     # teacher score를 학생 해상도에 맞춤
        #     S = resize(score_map, size=V.shape[-2:], mode='bilinear', align_corners=False)
        #     S_feat = self.score_adapter(S)
            
        #     # 학생 디코더 loss (pseudo로 감독)
        #     out_student = self.student_head_aspp([V, S_feat])
        #     loss_student = self.student_head_aspp.losses(out_student, gt_semantic_seg)
        #     for k, v in loss_student.items():
        #         losses[f'student_{k}'] = v

        # return losses

        # 0830 기준
        # [MOD] 학생 경로가 존재하면 '학생 손실'을 메인 decode.* 로 기록
        if self.use_student_head:
            feats_r50d = self.aux_backbone(img)                      # tuple of 4
            V = self.aux_aspp.forward_module(feats_r50d)             # (B,256,h,w)
            S = resize(score_map.detach(), size=V.shape[-2:],        # [MOD] detach 로 graident 차단
                       mode='bilinear', align_corners=False)
            S_feat = self.score_adapter(S)

            out_student = self.student_head_aspp([V, S_feat])
            loss_student = self.student_head_aspp.losses(out_student, gt_semantic_seg)
            losses.update(add_prefix(loss_student, 'decode'))        # [MOD] 핵심: decode.* = 학생

            # (선택) identity/aux 추가
            if self.with_identity_head:
                losses.update(self._identity_head_forward_train(score_map / self.tau, img_metas, gt_semantic_seg))

        else:
            # 학생 헤드가 없으면 teacher로 학습
            x_in = [text_embeddings] + x_orig if self.text_head else x_orig
            losses.update(self._decode_head_forward_train(x_in, img_metas, gt_semantic_seg))

            if self.with_identity_head:
                losses.update(self._identity_head_forward_train(score_map / self.tau, img_metas, gt_semantic_seg))
            if hasattr(self, 'auxiliary_head'):
                losses.update(self._auxiliary_head_forward_train(x_orig, img_metas, gt_semantic_seg))

        return losses
    
    # ------- inference 그대로 ------- 0821 기준
    # def encode_decode(self, img, img_metas):
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     x = self.extract_feat(img)

    #     _x_orig = [x[i] for i in range(4)]
    #     text_embeddings, x_orig, score_map = self.after_extract_feat(x)

    #     if self.with_neck:
    #         x_orig = list(self.neck(x_orig))

    #     if self.text_head:
    #         x = [text_embeddings,] + x_orig
    #     else:
    #         x = x_orig
    #     # print('text_embedding=', text_embeddings[0])
    #     out = self._decode_head_forward_test(x, img_metas)
    #     # print('cls_map=', out[0,:,40, 40])
        
    #     out = resize(
    #         input=out,
    #         size=img.shape[2:],
    #         mode='bilinear',
    #         align_corners=self.align_corners)
    #     return out

    # [MOD] encode_decode: 평가도 학생 출력 우선 사용 (0830 기준)
    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img)
        text_embeddings, x_orig, score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        if self.use_student_head:
            feats_r50d = self.aux_backbone(img)
            V = self.aux_aspp.forward_module(feats_r50d)
            # S = resize(score_map, size=V.shape[-2:], mode='bilinear', align_corners=False)
            S = resize(score_map, size=V.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            S_feat = self.score_adapter(S)
            out = self.student_head_aspp([V, S_feat])
        else:
            x_in = [text_embeddings] + x_orig if self.text_head else x_orig
            out = self._decode_head_forward_test(x_in, img_metas)

        out = resize(input=out, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return out
    
    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        if  torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred
