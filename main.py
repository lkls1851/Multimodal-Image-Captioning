from utils import *
from dataset import dataset
from trainer import Trainer

train_transforms=A.Compose([A.HorizontalFlip(), 
           A.RandomBrightnessContrast(),
           A.Resize(224,224),
           A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], always_apply=True),
           ToTensorV2()])

validation_transforms=A.Compose([A.Resize(224,224),
                                 A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], always_apply=True),
                                 ToTensorV2()])


tokeniser=GPT2TokenizerFast.from_pretrained('gpt2')
tokeniser.pad_token=tokeniser.eos_token


img_path='flickr_8k/Images'
annot_path='flickr_8k/captions.txt'


train_ds=dataset(annot_path=annot_path, img_path=img_path, transforms=train_transforms)
val_ds=dataset(annot_path=annot_path, img_path=img_path, transforms=validation_transforms)


def collate_fn(batch):
    img=[i[0] for i in batch]
    ids=[i[1] for i in batch]
    lab=[i[2] for i in batch]
    
    img=torch.stack(img, dim=0)
    ids=tokeniser.pad({'input_ids':ids}, padding='longest',
                     return_attention_mask=False,
                     return_tensors='pt')['input_ids']
    lab=tokeniser.pad({'input_ids':lab}, padding='longest',
                     return_attention_mask=False,
                     return_tensors='pt')['input_ids']
    return img, ids, lab

ds=dataset(annot_path=annot_path, img_path=img_path, transforms=train_transforms)
dataloader=DataLoader(dataset=ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

model_config = SimpleNamespace(
    vocab_size = 50_257,
    embed_dim = 768,
    num_heads = 12,
    seq_len = 1024,
    depth = 12,
    attn_dropout = 0.1,
    resid_dropout = 0.1,
    mlp_ratio = 4,
    mlp_dropout = 0.1,
    emb_dropout = 0.1,
)
train_config = SimpleNamespace(
    epochs = 5,
    freeze_epochs_gpt = 1,
    freeze_epochs_all = 2,
    lr = 1e-4,
    device = 'cuda',
    model_path = Path('captioner'),
    batch_size = 32
)

train_dl = torch.utils.data.DataLoader(train_ds,batch_size=train_config.batch_size,shuffle=True,
                                       pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds,batch_size=train_config.batch_size,shuffle=False,
                                     pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)

trainer = Trainer(model_config,train_config,(train_dl,val_dl))



