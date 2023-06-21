import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
import torchmetrics as tm

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import pytorch_lightning as pl

import re

import string


ARCHITECTURES = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base"
}

LABEL_NAMES = ['toxic', 'severe_toxic', 'obscene', 
               'threat', 'insult', 'identity_hate']


def plot_metrics(path: str) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    df = pd.read_csv(path)
    df = df.set_index('epoch')

    sns.lineplot(data=df[['train/loss', 'val/loss']], ax=ax[0])
    ax[0].set_ylabel('loss')
    sns.lineplot(data=df[['train/auroc', 'val/auroc']], ax=ax[1])
    ax[1].set_ylabel('auroc')
    sns.lineplot(data=df[['train/f1', 'val/f1']], ax=ax[2])
    ax[2].set_ylabel('f1 score')

    fig.tight_layout()
    plt.show()


def process_output(output_tensor: torch.tensor, threshold: float=0.5) -> list:
    test_pred = []

    for pred_batch in output_tensor:
        batch_list = (torch.nn.functional.sigmoid(pred_batch[1]) > threshold)\
        .int().tolist()

        for label in batch_list:
            test_pred.append(label)
    
    return test_pred


def remove_url(text):
    """Remove URLs from a sample string"""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_html(text):
    """Remove the html in sample text"""
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)


def remove_non_ascii(text):
    """Remove non-ASCII characters """
    return re.sub(r'[^\x00-\x7f]',r'', text)


def remove_special_characters(text):
    """Remove special special characters, including symbols, emojis, and 
    other graphic characters
    """
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_punctuation(text):
    """Remove the punctuation"""
    return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+|\n|\t', "", text)
    # return text.translate(str.maketrans('', '', string.punctuation))


def text_preprocessing(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df[text_col] = df[text_col].apply(lambda x: remove_url(x))
    df[text_col] = df[text_col].apply(lambda x: remove_html(x))
    df[text_col] = df[text_col].apply(lambda x: remove_non_ascii(x))
    df[text_col] = df[text_col].apply(lambda x: remove_special_characters(x))
    df[text_col] = df[text_col].apply(lambda x: remove_punctuation(x))

    return df


class ToxicCommentsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 arch: str="distilbert", 
                 max_sequence_length: int=256):
        self.tokenizer = AutoTokenizer.from_pretrained(ARCHITECTURES[arch], 
                                                       do_lower_case=True)
        
        self.max_sequence_length = max_sequence_length
        self.comment_types = LABEL_NAMES

        self.df = df[['comment_text', *self.comment_types]]
        self.df['label'] = list(self.df[self.comment_types].values)
        self.df = self.df.drop(columns=self.comment_types)

        #self.df['comment_text'] = self.df['comment_text'].str.strip()\
        #.str.replace('[\n\t=[\]{}<>|`^~\\\+/]', '', regex=True)

        self.df = text_preprocessing(self.df, 'comment_text')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        label = self.df['label'].iloc[idx]

        text_data = self.df['comment_text'].iloc[idx]

        encoded = self.tokenizer.encode_plus(
            text_data,
            add_special_tokens=True,
            max_length=self.max_sequence_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            pad_to_max_length=True,
            return_tensors='pt',
        )

        encoded = {k:v.squeeze(0) for k,v in encoded.items()}
        encoded['labels'] = torch.tensor(label).float()

        return encoded


class ToxicCommentClfModel(pl.LightningModule):
    def __init__(
            self, 
            class_names: list,
            train_df: pd.DataFrame=None,
            valid_df: pd.DataFrame=None,
            test_df: pd.DataFrame=None,
            arch: str="distilbert", 
            learning_rate: float=1e-3, 
            batch_size: int=32, 
            cm_period: int=5,
            max_sequence_length: int=256,
            class_weights: list=None,
            num_workers: int=2
        ):
        super().__init__()
        self.arch = arch

        self.num_classes = len(class_names)
        self.class_names = class_names

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        
        self.tf_model = AutoModelForSequenceClassification.from_pretrained(
            ARCHITECTURES[arch],
            problem_type='multi_label_classification',
            num_labels=self.num_classes,
            output_attentions=False,
            output_hidden_states=False
        )

        self.base_grad(requires_grad=False)

        self.lr = learning_rate

        self.batch_size = batch_size

        tm_kwargs = {
            'task':'multilabel',
            'num_labels':self.num_classes,
            'compute_on_step':False
        }

        self.train_auroc = tm.AUROC(**tm_kwargs)
        self.val_auroc = tm.AUROC(**tm_kwargs)
        self.test_auroc = tm.AUROC(**tm_kwargs)
        
        self.train_f1 = tm.F1Score(**tm_kwargs)
        self.val_f1 = tm.F1Score(**tm_kwargs)
        self.test_f1 = tm.F1Score(**tm_kwargs)
        
        self.val_cm = tm.ConfusionMatrix(**tm_kwargs)
        self.test_cm = tm.ConfusionMatrix(**tm_kwargs)

        self.cm_period = cm_period

        self.max_seq_len = max_sequence_length

        self.class_weights = class_weights

        self.num_workers = num_workers
    
    def base_grad(self, requires_grad: bool):
        for param in self.tf_model.__getattr__(self.arch).parameters():
            param.requires_grad = requires_grad

    def forward(self, batch):
        out = self.tf_model(**batch)

        if self.class_weights is not None:
            criterion = torch.nn.BCEWithLogitsLoss(
                weight=torch.tensor(self.class_weights).to(self.device))
            loss = criterion(out.logits, batch['labels'])
        else:
            loss = out.loss

        return loss, out.logits, out.attentions
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], 
            lr=self.lr, eps=1e-08)
    
    def train_dataloader(self):
        dataset = ToxicCommentsDataset(self.train_df, arch=self.arch, 
                                       max_sequence_length=self.max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader

    def val_dataloader(self):
        dataset = ToxicCommentsDataset(self.valid_df, arch=self.arch, 
                                       max_sequence_length=self.max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader

    def test_dataloader(self):
        dataset = ToxicCommentsDataset(self.test_df, arch=self.arch, 
                                       max_sequence_length=self.max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader
    
    def predict_dataloader(self):
        dataset = ToxicCommentsDataset(self.test_df, arch=self.arch, 
                                       max_sequence_length=self.max_seq_len)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return loader
        
    def training_step(self, batch, batch_idx):
        loss, logit, _ = self.forward(batch)
        
        self.train_auroc.update(logit, batch['labels'].int())
        self.train_f1.update(logit, batch['labels'].int())
        self.log_dict(
            {
                'train/loss': loss, 
                'train/auroc': self.train_auroc, 
                'train/f1': self.train_f1
            }, 
            on_epoch=True, 
            on_step=False
        )
        
        return loss

    def on_train_epoch_end(self):
        self.train_f1.reset()
        self.train_auroc.reset()

        self.val_f1.reset()
        self.val_cm.reset()
        self.val_auroc.reset()
    
    def validation_step(self, batch, batch_idx):
        loss, logit, _ = self.forward(batch)

        self.val_auroc.update(logit, batch['labels'].int())
        self.val_f1.update(logit, batch['labels'].int())
        self.log_dict(
            {
                'val/loss': loss, 
                'val/auroc': self.val_auroc,
                'val/f1': self.val_f1
            }
        )

        self.val_cm.update(logit, batch['labels'].int())

    def plot_confusion_matrix(self, df):
        plt.figure(figsize=(11,4))
        ax = sns.heatmap(df, annot=True, cmap='magma', fmt='')
        ax.set_title(f'Confusion Matrix (Epoch {self.current_epoch+1})')
        ax.set_ylabel('True labels')
        ax.set_xlabel('Predicted labels')
        plt.show()

    def plot_multilabel_cm(self, cms: np.ndarray):
        fig, ax = plt.subplots(2, 3, figsize=(10,6))
        curr_cm = 0

        for i in range(2):
            for j in range(3):
                df = pd.DataFrame(cms[curr_cm])
                sns.heatmap(df, annot=True, cmap='magma', fmt='', ax=ax[i][j])
                ax[i][j].set_title(f'Label {self.class_names[curr_cm]} CM')
                ax[i][j].set_ylabel('True labels')
                ax[i][j].set_xlabel('Predicted labels')
                curr_cm += 1

        fig.suptitle(
            f'Multilabel Confusion Matrix Grid (Epoch {self.current_epoch+1})')
        fig.tight_layout()
        plt.show()

    def on_validation_epoch_end(self):
        if self.current_epoch>0 and (self.current_epoch+1)%self.cm_period==0:
            self.plot_multilabel_cm(
                self.val_cm.compute().detach().cpu().numpy().astype(int))

    def test_step(self, batch, batch_idx):
        loss, logit, _ = self.forward(batch)

        self.test_auroc.update(logit, batch['labels'].int())
        self.test_f1.update(logit, batch['labels'].int())
        self.log_dict(
            {
                'test/loss': loss, 
                'test/auroc': self.test_auroc,
                'test/f1': self.test_f1
            }
        )

        self.test_cm.update(logit, batch['labels'].int())

    def on_test_epoch_end(self):
        self.plot_multilabel_cm(
            self.test_cm.compute().detach().cpu().numpy().astype(int))

    def predict_step(self, batch, batch_idx):
        return self(batch)