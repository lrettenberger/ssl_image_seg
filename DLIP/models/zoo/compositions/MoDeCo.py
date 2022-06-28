

from DLIP.models.zoo.compositions.moco_v2 import Mocov2
from DLIP.models.zoo.encoder.resnet_encoder import ResNetEncoder
from torch import nn
import torch
from torch.nn import functional as F


from DLIP.models.zoo.necks.mo_de_co_neck import MoDeCoNeck

class MoDeCo(Mocov2):
    
    def __init__(
        self,
        base_encoder='resnet50',
        emb_dim: int = 128, 
        num_negatives: int = 34607,
        num_negatives_val: int = 8655,
        encoder_momentum: float = 0.999, 
        softmax_temperature: float = 0.07, 
        neck='moco'
    ):
        super().__init__(base_encoder, emb_dim, num_negatives, num_negatives_val, encoder_momentum, softmax_temperature, neck)
        self.encoder_q = ResNetEncoder(
                        input_channels = 3,
                        encoder_type = 'resnet50',
                        pretraining_weights=None,
                        encoder_frozen=False
                    )
        self.encoder_k = ResNetEncoder(
                        input_channels = 3,
                        encoder_type = 'resnet50',
                        pretraining_weights=None,
                        encoder_frozen=False
                    )
        dim_mlp = self.encoder_q.backbone[-1][-1].bn3.weight.shape[0]
        self.encoder_q = nn.Sequential(self.encoder_q, MoDeCoNeck(dim_mlp=dim_mlp,emb_dim=emb_dim))
        self.encoder_k = nn.Sequential(self.encoder_k, MoDeCoNeck(dim_mlp=dim_mlp,emb_dim=emb_dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # create the queues


        # train

        self.register_buffer("queue_global", torch.randn(emb_dim, num_negatives))
        self.queue_global = nn.functional.normalize(self.queue_global, dim=0)
        self.register_buffer("queue_ptr_global", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_x1", torch.randn(emb_dim, num_negatives))
        self.queue_x1 = nn.functional.normalize(self.queue_x1, dim=0)
        self.register_buffer("queue_ptr_x1", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_x2", torch.randn(emb_dim, num_negatives))
        self.queue_x2 = nn.functional.normalize(self.queue_x2, dim=0)
        self.register_buffer("queue_ptr_x2", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_x3", torch.randn(emb_dim, num_negatives))
        self.queue_x3 = nn.functional.normalize(self.queue_x3, dim=0)
        self.register_buffer("queue_ptr_x3", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_x4", torch.randn(emb_dim, num_negatives))
        self.queue_x4 = nn.functional.normalize(self.queue_x4, dim=0)
        self.register_buffer("queue_ptr_x4", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_x5", torch.randn(emb_dim, num_negatives))
        self.queue_x5 = nn.functional.normalize(self.queue_x5, dim=0)
        self.register_buffer("queue_ptr_x5", torch.zeros(1, dtype=torch.long))


        # val
        self.register_buffer("val_queue_global", torch.randn(emb_dim, num_negatives_val))
        self.val_queue_global = nn.functional.normalize(self.val_queue, dim=0)
        self.register_buffer("val_queue_ptr_global", torch.zeros(1, dtype=torch.long))

        self.register_buffer("val_queue_x1", torch.randn(emb_dim, num_negatives_val))
        self.val_queue_x1 = nn.functional.normalize(self.val_queue_x1, dim=0)
        self.register_buffer("val_queue_ptr_x1", torch.zeros(1, dtype=torch.long))

        self.register_buffer("val_queue_x2", torch.randn(emb_dim, num_negatives_val))
        self.val_queue_x2 = nn.functional.normalize(self.val_queue_x2, dim=0)
        self.register_buffer("val_queue_ptr_x2", torch.zeros(1, dtype=torch.long))

        self.register_buffer("val_queue_x3", torch.randn(emb_dim, num_negatives_val))
        self.val_queue_x3 = nn.functional.normalize(self.val_queue_x3, dim=0)
        self.register_buffer("val_queue_ptr_x3", torch.zeros(1, dtype=torch.long))

        self.register_buffer("val_queue_x4", torch.randn(emb_dim, num_negatives_val))
        self.val_queue_x4 = nn.functional.normalize(self.val_queue_x4, dim=0)
        self.register_buffer("val_queue_ptr_x4", torch.zeros(1, dtype=torch.long))

        self.register_buffer("val_queue_x5", torch.randn(emb_dim, num_negatives_val))
        self.val_queue_x5 = nn.functional.normalize(self.val_queue_x5, dim=0)
        self.register_buffer("val_queue_ptr_x5", torch.zeros(1, dtype=torch.long))


        
    def forward(self, img_q, img_k, queues):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queues: queues from which to pick negative samples
        Output:
            logits, targets
        """

        # compute query features
        q, layer_qs = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k, layer_ks = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        
        # global
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, queues[0].clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.softmax_temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # local
        l_pos_x1 = torch.einsum("nc,nc->n", [layer_qs[0], layer_ks[0]]).unsqueeze(-1)
        l_neg_x1 = torch.einsum("nc,ck->nk", [layer_qs[0], queues[1].clone().detach()])
        logits_x1 = torch.cat([l_pos_x1, l_neg_x1], dim=1)
        logits_x1 /= self.softmax_temperature
        labels_x1 = torch.zeros(logits_x1.shape[0], dtype=torch.long)
        labels_x1 = labels.type_as(logits_x1)

        l_pos_x2 = torch.einsum("nc,nc->n", [layer_qs[1], layer_ks[1]]).unsqueeze(-1)
        l_neg_x2 = torch.einsum("nc,ck->nk", [layer_qs[1], queues[2].clone().detach()])
        logits_x2 = torch.cat([l_pos_x2, l_neg_x2], dim=1)
        logits_x2 /= self.softmax_temperature
        labels_x2 = torch.zeros(logits_x2.shape[0], dtype=torch.long)
        labels_x2 = labels.type_as(logits_x2)

        l_pos_x3 = torch.einsum("nc,nc->n", [layer_qs[2], layer_ks[2]]).unsqueeze(-1)
        l_neg_x3 = torch.einsum("nc,ck->nk", [layer_qs[2], queues[3].clone().detach()])
        logits_x3 = torch.cat([l_pos_x3, l_neg_x3], dim=1)
        logits_x3 /= self.softmax_temperature
        labels_x3 = torch.zeros(logits_x3.shape[0], dtype=torch.long)
        labels_x3 = labels.type_as(logits_x3)

        l_pos_x4 = torch.einsum("nc,nc->n", [layer_qs[3], layer_ks[3]]).unsqueeze(-1)
        l_neg_x4 = torch.einsum("nc,ck->nk", [layer_qs[3], queues[4].clone().detach()])
        logits_x4 = torch.cat([l_pos_x4, l_neg_x4], dim=1)
        logits_x4 /= self.softmax_temperature
        labels_x4 = torch.zeros(logits_x4.shape[0], dtype=torch.long)
        labels_x4 = labels.type_as(logits_x4)
        
        l_pos_x5 = torch.einsum("nc,nc->n", [layer_qs[4], layer_ks[4]]).unsqueeze(-1)
        l_neg_x5 = torch.einsum("nc,ck->nk", [layer_qs[4], queues[5].clone().detach()])
        logits_x5 = torch.cat([l_pos_x5, l_neg_x5], dim=1)
        logits_x5 /= self.softmax_temperature
        labels_x5 = torch.zeros(logits_x5.shape[0], dtype=torch.long)
        labels_x5 = labels.type_as(logits_x5)

        logits = [logits,logits_x1,logits_x2,logits_x3,logits_x4,logits_x5]
        labels = [labels,labels_x1,labels_x2,labels_x3,labels_x4,labels_x5]
        k = [k]+layer_ks

        return logits, labels, k
    
    
    def training_step(self, batch, batch_idx):
        (img_1,img_2), (_) = batch

        #self._momentum_update_key_encoder()  # update the key encoder
        output, target, keys = self(img_q=img_1, img_k=img_2, queues=[self.queue_global,self.queue_x1,self.queue_x2,self.queue_x3,self.queue_x4,self.queue_x5])

        losses = []
        
        self._dequeue_and_enqueue(keys[0], queue=self.queue_global, queue_ptr=self.queue_ptr_global)
        losses.append(F.cross_entropy(output[0].float(), target[0].long()))

        self._dequeue_and_enqueue(keys[1], queue=self.queue_x1, queue_ptr=self.queue_ptr_x1)
        losses.append(F.cross_entropy(output[1].float(), target[1].long()))

        self._dequeue_and_enqueue(keys[2], queue=self.queue_x2, queue_ptr=self.queue_ptr_x2)
        losses.append(F.cross_entropy(output[2].float(), target[2].long()))

        self._dequeue_and_enqueue(keys[3], queue=self.queue_x3, queue_ptr=self.queue_ptr_x3)
        losses.append(F.cross_entropy(output[3].float(), target[3].long()))

        self._dequeue_and_enqueue(keys[4], queue=self.queue_x4, queue_ptr=self.queue_ptr_x4)
        losses.append(F.cross_entropy(output[4].float(), target[4].long()))

        self._dequeue_and_enqueue(keys[5], queue=self.queue_x5, queue_ptr=self.queue_ptr_x5)
        losses.append(F.cross_entropy(output[5].float(), target[5].long()))

        loss = sum(losses)

        self.log("train/loss_global",losses[0],  prog_bar=True, on_epoch=True)
        self.log("train/loss_1",losses[1],  prog_bar=True, on_epoch=True)
        self.log("train/loss_2",losses[2],  prog_bar=True, on_epoch=True)
        self.log("train/loss_3",losses[3],  prog_bar=True, on_epoch=True)
        self.log("train/loss_4",losses[4],  prog_bar=True, on_epoch=True)
        self.log("train/loss_5",losses[5],  prog_bar=True, on_epoch=True)
        self.log("train/loss",loss,  prog_bar=True, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        (img_1,img_2), (_) = batch

        #self._momentum_update_key_encoder()  # update the key encoder
        output, target, keys = self(img_q=img_1, img_k=img_2, queues=[self.val_queue_global,self.val_queue_x1,self.val_queue_x2,self.val_queue_x3,self.val_queue_x4,self.val_queue_x5])

        losses = []
        
        self._dequeue_and_enqueue(keys[0], queue=self.val_queue_global, queue_ptr=self.val_queue_ptr_global,val_step=True)
        losses.append(F.cross_entropy(output[0].float(), target[0].long()))

        self._dequeue_and_enqueue(keys[1], queue=self.val_queue_x1, queue_ptr=self.val_queue_ptr_x1,val_step=True)
        losses.append(F.cross_entropy(output[1].float(), target[1].long()))

        self._dequeue_and_enqueue(keys[2], queue=self.val_queue_x2, queue_ptr=self.val_queue_ptr_x2,val_step=True)
        losses.append(F.cross_entropy(output[2].float(), target[2].long()))

        self._dequeue_and_enqueue(keys[3], queue=self.val_queue_x3, queue_ptr=self.val_queue_ptr_x3,val_step=True)
        losses.append(F.cross_entropy(output[3].float(), target[3].long()))

        self._dequeue_and_enqueue(keys[4], queue=self.val_queue_x4, queue_ptr=self.val_queue_ptr_x4,val_step=True)
        losses.append(F.cross_entropy(output[4].float(), target[4].long()))

        self._dequeue_and_enqueue(keys[5], queue=self.val_queue_x5, queue_ptr=self.val_queue_ptr_x5,val_step=True)
        losses.append(F.cross_entropy(output[5].float(), target[5].long()))

        loss = sum(losses)

        self.log("val/loss_global",losses[0],  prog_bar=True, on_epoch=True)
        self.log("val/loss_1",losses[1],  prog_bar=True, on_epoch=True)
        self.log("val/loss_2",losses[2],  prog_bar=True, on_epoch=True)
        self.log("val/loss_3",losses[3],  prog_bar=True, on_epoch=True)
        self.log("val/loss_4",losses[4],  prog_bar=True, on_epoch=True)
        self.log("val/loss_5",losses[5],  prog_bar=True, on_epoch=True)
        self.log("val/loss",loss,  prog_bar=True, on_epoch=True)

        return loss