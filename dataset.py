from utils import *

class dataset:
    def __init__ (self, annot_path, img_path, transforms):
        self.df=pd.read_csv(annot_path)
        self.df['image']=self.df['image'].apply(lambda x: os.path.join(img_path, x))
        self.transforms=transforms
        self.tokeniser=GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokeniser.pad_token=self.tokeniser.eos_token
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row=self.df.iloc[index,:]
        img=row['image']
        cap=row['caption']
        img=Image.open(img).convert('RGB')
        img=np.array(img)
        augs=self.transforms(image=img)
        img=augs['image']
        cap=f'{cap}<|endoftext|>'
        ids=self.tokeniser.encode_plus(cap, truncation=True)['input_ids']
        labels=ids.copy()
        labels[:-1]=ids[1:] ## To remove the eos padding from labels
        return img, ids, labels