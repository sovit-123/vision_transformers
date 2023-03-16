import torch
import torch.nn as nn

class DETRModel(nn.Module):
    def __init__(self, num_classes=1, model='detr_resnet50'):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.model = torch.hub.load(
            'facebookresearch/detr', 
            model, 
            pretrained=True,
        )
        self.out = nn.Linear(
            in_features=self.model.class_embed.out_features, 
            out_features=num_classes
        )
        
    def forward(self, images):
        d = self.model(images)
        d['pred_logits'] = self.out(d['pred_logits'])
        return d
    
    def parameter_groups(self):
        return { 
            'backbone': [p for n,p in self.model.named_parameters()
                              if ('backbone' in n) and p.requires_grad],
            'transformer': [p for n,p in self.model.named_parameters() 
                                 if (('transformer' in n) or ('input_proj' in n)) and p.requires_grad],
            'embed': [p for n,p in self.model.named_parameters()
                                 if (('class_embed' in n) or ('bbox_embed' in n) or ('query_embed' in n)) 
                           and p.requires_grad],
            'final': self.out.parameters()
        }